## Before training

Create image list by running "/border-legibility/bing_maps/create_image_list.py" with the name of the scrape directory as the first command 
line argument. (Ex: $ python3 create_image_list.py global_scrape)

## Training
Training is meant to be as simple as:
` python train.py config.json `

This uses the SiameseDecider model defined in`model.py`, the hyper parameters defined in`config.json`, and random images from our dataset to train and log results via wandb and pytorch lightning.

The results of the training are saved in `./weights` with the name defined in `config.json`. 
Make sure to set up the correct file path in `config.json`.

If you want to train but disable logging, do:
`python train.py config.json no`

## Using a trained model
Given the string paths to two images (x1, x2) and a path to saved weights (weight_path):
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseDecider.load_from_checkpoint(weight_path,
                                                      						 batch_size=4, map_location=device)

# Create images
og_x1, og_x2 = [bu.imread(x) for x in [x1, x2]]
h, w, c = og_x1.shape

# Create masks
mask_1, mask_2 = [
	create_tri_mask(x, (h,w), (h,w), color=2, thick=5, show=False)
	for x in [x1, x2]
]

x1 = model.trainloader.dataset.prep_img(og_x1, rot=False).unsqueeze(0).to(device)
x2 = model.trainloader.dataset.prep_img(og_x2, rot=False).unsqueeze(0).to(device)

# Map images to prediction list: (x1, x2) -> [0.1, 0.9]
y_hat = model(x1, x2).cpu().squeeze().softmax(dim=0).detach().tolist()

```