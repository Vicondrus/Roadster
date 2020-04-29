import numpy as np
from skimage import exposure
from skimage import transform


def vote_on_image(models, image):
    image = transform.resize(image, (32, 32))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    voting_dict = {}
    for voter in models:
        preds = voter.predict(image)
        top = np.argsort(-preds, axis=1)
        for i, vote in enumerate(top[0][:3]):
            if vote not in voting_dict:
                voting_dict[vote] = 3 - i
            else:
                voting_dict[vote] += 3 - i

    winner = max(voting_dict, key=voting_dict.get)

    return winner
