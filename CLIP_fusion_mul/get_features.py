import clip
import torch



device = torch.device("cuda")


def features(image_data,text_data,clip_model):
    clip_model.to(device)
    image_features_set=[]
    text_features_set=[]
    with torch.no_grad():
        for text_trains, image_trains in zip(text_data,image_data):
            image_features, labels = image_trains
            image_features = clip_model.encode_image(image_features.to(device))
            text_token = clip.tokenize(text_trains).to(device)
            text_feature = clip_model.encode_text(text_token)
            image_features_set.extend(image_features)
            text_features_set.extend(text_feature)
    return text_features_set,image_features_set
