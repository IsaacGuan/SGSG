# SGSG: Semantics-Guided Shape Generation

This repository contains the implementation of the label regression network (LRN) and the web application of the semantics-guided shape generator described in the paper ["Semantics-Guided Latent Space Exploration for Shape Generation"](https://diglib.eg.org/handle/10.1111/cgf142619). For the implementations of the shape encoder network (SEN) and the shape decoder network (SDN) also described in the paper, please refer to [IsaacGuan/3D-GAE](https://github.com/IsaacGuan/3D-GAE) and [IsaacGuan/implicit-decoder/IMGAN](https://github.com/IsaacGuan/implicit-decoder/tree/master/IMGAN), respectively.

## Training LRN

Run the following commands.

```bash
python train.py --dataset_name chairs
python train.py --dataset_name lamps
python train.py --dataset_name tables
```

## Testing LRN and Generating the Gaussians

Run the following commands.

```bash
python test.py --dataset_name chairs
python test.py --dataset_name lamps
python test.py --dataset_name tables
```

## Running the Web Application

In the `web-app` folder, run the following command and then visit http://localhost:5000/ from your browser.

```bash
python app.py
```
