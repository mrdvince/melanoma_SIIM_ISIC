import os
from flask import Flask, request, render_template

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SEResNext50(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResNext50, self).__init__()
        self.models = pretrainedmodels.__dict__[
            'se_resnext50_32x4d'](pretrained=pretrained)
        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.models.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))
        return out, loss


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_file.save(os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            ))
            return render_template('index.html', prediction=1)

    return render_template('index.html', prediction=0)


if __name__ == '__main__':
    app.run(debug=True)
