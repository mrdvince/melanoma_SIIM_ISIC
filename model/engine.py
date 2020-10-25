import torch
from tqdm import tqdm
from average_meter import AverageMeter

try:
    import torch_xla.core.xla_model as xm

    __xla_available = True
except ImportError:
    __xla_available = False

try:
    from apex import amp

    __apex_available = True
except ImportError:
    __apex_available = False


class Engine:
    @staticmethod
    def train(
        data_loader,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1,
        use_tpu=False,
        fp16=False,
    ):
        if use_tpu and __xla_available:
            raise Exception(
                "You want to use TPUs but you dont have pytorch_xla installed"
            )
        if fp16 and __apex_available:
            raise Exception("You want to use fp16 but you dont have apex installed")
        if fp16 and use_tpu:
            raise Exception("Apex fp16 is not available when using TPUs")
        if fp16:
            accumulation_steps = 1
        losses = AverageMeter()
        predictions = []
        model.train()
        if accumulation_steps > 1:
            optimizer.zero_grad()
        tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            if accumulation_steps == 1 and b_idx == 0:
                optimizer.zero_grad()
            _, loss = model(**data)

            if not use_tpu:
                with torch.set_grad_enabled(True):
                    if fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    if (b_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        if b_idx > 0:
                            optimizer.zero_grad()
            else:
                loss.backward()
                xm.optimizer_step(optimizer)
                if scheduler is not None:
                    scheduler.step()
                if b_idx > 0:
                    optimizer.zero_grad()

            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg)
        return losses.avg

    @staticmethod
    def evaluate(data_loader, model, device, use_tpu=False):
        losses = AverageMeter()
        final_predictions = []
        model.eval()
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, loss = model(**data)
                predictions = predictions.cpu()
                losses.update(loss.item(), data_loader.batch_size)
                final_predictions.append(predictions)
                tk0.set_postfix(loss=losses.avg)
        return final_predictions, losses.avg

    @staticmethod
    def predict(data_loader, model, device, use_tpu=False):
        model.eval()
        final_predictions = []
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, _ = model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        return final_predictions