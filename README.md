# Dense-U-net-Tensorflow
Simple Tensorflow implementation of ["Dense-U-net: Dense encoder-decoder network for holographic imaging of 3D particle field" ]

## Usage
### datasets
* The datasets is generated through Layer-oriented-algorithm-and-ground-true / creat_train_datasets.m.(use MATLAB)
* Layer-oriented-algorithm-and-ground-true/datasets used to store training datasets and test datasets
* For `your dataset`, put images like this:

```
├── dataset
   └── YOUR_DATASET_NAME (input data)
       ├── YOUR_DATASET_NAME 
           ├── YOUR_DATASET_NAME
       	        ├──xxx.tif (name, format doesnot matter)
	            ├──yyy.tif
	            └── ...
    └── YOUR_DATASET_NAME (ground true)
       ├── YOUR_DATASET_NAME 
           ├── YOUR_DATASET_NAME
       	        ├──xxx.tif (name, format doesnot matter)
	            ├──yyy.tif
	            └── ...
```
### train
* Dense-U-net
```
Replace the data set address of "train_generator" in Dense-U-net /Dense-U-net.py with your own path. 
Change the location of the loss value of "plot_history" to your own address.
Set "is_train" under "if __name__ == '__main__':" to True.
Python Dense-U-net.py can train the Dense-U-net network
```

### test
* Dense-U-net
```
Replace the data set address of "train_generator" in Dense-U-net /Dense-U-net.py with your own path. 
Change the location of the loss value of "plot_history" to your own address.
Set "is_train" under "if __name__ == '__main__':" to False.
"Cv2.imread" reads the data into its own address,
Python Dense-U-net.py can train the Dense-U-net network
```

### particles_information_extraction
* Replace "cv2.imread" in the for loop with the address of the particle image you want to extract.
* python particles_information_extraction.py   can extract particle information

## Author
Yufeng Wu（cjluyufengwu@163.com); Liangcai Cao(clc@tsinghua.edu.cn)
