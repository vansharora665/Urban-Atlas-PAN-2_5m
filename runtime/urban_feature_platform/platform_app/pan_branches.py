"""A fixed learned water branch and a separately adapted land decoder."""

import torch


class PanWaterLand(torch.nn.Module):
    def __init__(self,water_model,land_model):
        super().__init__()
        for name,value in water_model.encoder.state_dict().items():
            if not torch.equal(value,land_model.encoder.state_dict()[name]):
                raise ValueError("Shared PAN decoder branches must have identical encoder weights")
        self.encoder=water_model.encoder
        self.water_decoder=water_model.decoder
        self.water_head=water_model.segmentation_head
        self.land_decoder=land_model.decoder
        self.land_head=land_model.segmentation_head

    def forward(self,image):
        features=self.encoder(image)
        water_logits=self.water_head(self.water_decoder(features))
        land_logits=self.land_head(self.land_decoder(features))
        # The fixed water branch supplies its probability mass. A conditional
        # land decoder distributes the remainder; neither head sets an area quota.
        water=water_logits.softmax(1)[:,:1]
        land=land_logits[:,1:].softmax(1)
        return torch.cat((water,(1-water)*land),dim=1).clamp_min(1e-8).log()
