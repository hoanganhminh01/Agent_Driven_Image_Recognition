import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FeatureExtractor(nn.Module):
    def __init__(self, network="vgg16"):
        super(FeatureExtractor, self).__init__()
        if network[:5] == "vgg16":
            model = torchvision.models.vgg16(pretrained=True)
        elif network[:5] == "vgg19":
            model = torchvision.models.vgg19(pretrained=True)
        elif network[:8] == "resnet50":
            model = torchvision.models.resnet50(pretrained=True)
        elif network[:9] == "resnet152":
            model = torchvision.models.resnet152(pretrained=True)
        elif network[:11] == "inceptionv3":
            model = torchvision.models.inception_v3(pretrained=True)
        elif network[:11] == "densenet121":
            model = torchvision.models.densenet121(pretrained=True)
        elif "efficientnet_b0" in network:
            model = torchvision.models.efficientnet_b0(pretrained=True)
        else:
            model = torchvision.models.alexnet(pretrained=True)
        model.eval()  # to not do dropout
        if "vgg" in network:
            self.features = list(model.children())[0]
            self.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
        elif "resnet" in network:
            self.features = nn.Sequential(*list(model.children())[:-2])
            print(self.features)

            self.classifier = nn.Sequential(
                nn.Linear(model.fc.in_features, model.fc.out_features),
                nn.Softmax(dim=1),
            )
        elif "inceptionv3" in network:
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.classifier = nn.Linear(model.fc.in_features, model.fc.out_features)
        elif "densenet" in network:
            self.features = model.features
            # self.classifier = nn.Linear(model.classifier.in_features, model.classifier.out_features)
        else:  # for AlexNet
            self.features = model.features
            self.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x


class DQN(nn.Module):
    def __init__(self, h, w, outputs, history_length, model_name):
        super(DQN, self).__init__()
        if "vgg" in model_name:
            feat_dim = outputs * history_length + 25088
        elif "resnet" in model_name:
            feat_dim = outputs * history_length + 100352
        elif "densenet" in model_name:
            feat_dim = outputs * history_length + 9441 - 225

        # print(outputs * history_length , model_name)
        # print(feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=outputs),
        )

    def forward(self, x):
        return self.classifier(x)


class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPO, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))
        return self.actor_linear(x), self.critic_linear(x)
