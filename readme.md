# OmniCityMapping

## 1. Introduction
This repo is the official Mmdetection implementation of Fine-grained Building Attribute Mapping Based on Street-view Images and A Cross-view Matching Method, includes a part of the dataset and benchmarks of building attributes recognition networks. The whole dataset and the code of cross-view matching algorithm will be released later.

## 2. Datasets and annotations
The dataset was divided into [train](https://drive.google.com/file/d/1yiGiDs0U1z8eRyiTAh1lqXOThHjZ0-Jm/view) and [validation](https://drive.google.com/file/d/1Yu_P-gDxtdSRWyKoWBmlQ4PBRhYftf4g/view) set, which includes annotations for four building attributes (i.e. land use, number of floors, year built and floor area ratio), respectively:
Attribute|train/val|Annotation
----|----------|----------
Land use|train|[annotation-train-landuse](https://drive.google.com/file/d/1zvNJQb49_9K1TaUN9Nyj2XDFAaSZiSVG/view?usp=drive_link)
Land use|val|[annotation-val-landuse](https://drive.google.com/file/d/18DiqsXreeIGPnxuWxRI_rgorCbWTz5eP/view?usp=drive_link)
Number of floors|train|[annotation-train-floor](https://drive.google.com/file/d/1yghMHSQLHbRDmt7dm_cuZrSizF1dQUA9/view?usp=drive_link)
Number of floors|val|[annotation-val-floor](https://drive.google.com/file/d/1eMeh6-7rJZDESNduLm4JuZ5ES-V6Ms6y/view?usp=drive_link)
Year built|train|[annotation-train-yearbuilt](https://drive.google.com/file/d/1_bHDoW-VIiu0r5u7U4_Au6oZ-EHGpf6x/view?usp=drive_link)
Year built|val|[annotation-val-yearbuilt](https://drive.google.com/file/d/15vymdWt0udaL5th_w48uyUDMl4LK_tAe/view?usp=drive_link)
Floor area ratio|train|[annotation-train-FAR](https://drive.google.com/file/d/1iDOsWjP8akIFsqQFziX2_xLqsOINTfO6/view?usp=drive_link)
Floor area ratio|val|[annotation-val-FAR](https://drive.google.com/file/d/11cg8Bkrg_EZTE83fLMEswA1L3FwuLGK4/view?usp=drive_link)

## 3. Overview of the models and its related tasks
Data Type: Panorama image with a size of 2048*1024
Tasks: Land use detection/segmentation, Number of floors detection/segmentation, Year built detection/segmentation and Floor area ratio detection/segmentation
### Land use
Method | $mAP$ | $mAP_{50}$ | Task | config | Download
-------|-------|-------|---------|-----|--------|
RetinaNet|0.282|0.385|Object detection|[config](https://drive.google.com/file/d/1-aTuv7P74OnYz6n8GkPB6YRGH7wO_DL5/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1bEawj0DnnybtIjBvYGJrxTRJj7KnRUmI/view?usp=drive_link)
YOLO-F|0.248|0.378|Object detection|[config](https://drive.google.com/file/d/16WC3gBb5b1iiQTfpB2poCXLIMV18w5Xg/view?usp=drive_link)|[Model](https://drive.google.com/file/d/17b5Cidv9zpaQlbXVV96HOPHAKgmeXNa1/view?usp=drive_link)
TOOD|0.290|0.373|Object detection|[config](https://drive.google.com/file/d/1V5SUgsuCVwiHVG1ymmPEQRpI8gchP5Wq/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1rkCywjZHx82y6-b6-VYN1YPcQxeR8j5-/view?usp=drive_link)
Faster R-CNN|0.276|0.381|Object detection|[config](https://drive.google.com/file/d/1Rs6BCMORSZqylZM5L-rirVceOdF1Ge62/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1fPil4bVvVsIyuj-T8ggdl7qK7lGlSkuZ/view?usp=drive_link)
Mask R-CNN|0.281|0.382|Object detection & Instance segmentation|[config](https://drive.google.com/file/d/1TxB1HTQ2sSh4t5U9Rvg3KNLi-cpV17a-/view?usp=drive_link)|[Model](https://drive.google.com/file/d/19Amz7cyStSpxzEaUWuTuYNoxtXv85xa5/view?usp=drive_link)
Cascade R-CNN|0.286|0.378|Object detectionn|[config](https://drive.google.com/file/d/1zsnXQ-khDBk0-lIDmzjdYLZC9UfeXLQZ/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1IjQQKSpBBa5xDPVs3Q8jr8Do-gH14Eh5/view?usp=drive_link)
RetinaNet (Swin)|0.291|0.400|Object detection|[config](https://drive.google.com/file/d/19afFchNwLvozrFRg6gIK1rOOwA6ovorQ/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1YfByTDi2lFdvzKtUUpK9OyWV0XqurxT2/view?usp=drive_link)
Mask R-CNN (Swin)|0.274|0.367|Object detection & Instance segmentation|[config](https://drive.google.com/file/d/19afFchNwLvozrFRg6gIK1rOOwA6ovorQ/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1YfByTDi2lFdvzKtUUpK9OyWV0XqurxT2/view?usp=drive_link)
PVT v2|0.265|0.358|Object detection|[config](https://drive.google.com/file/d/1aQZ839DHMuOmD1EpKOM9G-2rzvqo8zgF/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1lE7Qp44Muggs62WIpy5qmPuHuRvajQKu/view?usp=drive_link)

### Number of floors
Method | $mAP$ | $mAP_{50}$ | Task | config | Download
-------|-------|-------|---------|-----|--------|
RetinaNet|0.332|0.450|Object detection|[config](https://drive.google.com/file/d/1kyeQ8aycCVE4ussNVEbhEt_MwZE0K60P/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1noOvRZnKhJFALpdjDQm4Xn_D6iK1Zvct/view?usp=drive_link)
YOLO-F|0.362|0.559|Object detection|[config](https://drive.google.com/file/d/1iAMLcE2XPzaRWdh_iBq893WnqpG1Px9N/view?usp=drive_link)|[Model](https://drive.google.com/file/d/19mLanzOpOjAv4K0vN29MvVZwWKezrGyr/view?usp=drive_link)
TOOD|0.299|0.398|Object detection|[config](https://drive.google.com/file/d/1F28PrPt67HBMPUCPqqPLCnQ-I-tnLGgO/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1CiRxPsVLqX8NRIHW3O6GIAEon60ieBPI/view?usp=drive_link)
Faster R-CNN|0.346|0.482|Object detection|[config](https://drive.google.com/file/d/1bux3gU6thiW9FOWZCtGDJfHIvggUwUjq/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1eb0XGwXQSanvnqODDB3oWTGHWjYjRoLa/view?usp=drive_link)
Mask R-CNN|0.352|0.482|Object detection & Instance segmentation|[config](https://drive.google.com/file/d/1Ju7yA1OlJ5UBAB87KVOS2_cBZJE_kRTi/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1jShls2-RkIA8rnk9CCyV3TYWJTsh1_7C/view?usp=drive_link)
Cascade R-CNN|0.360|0.477|Object detectionn|[config](https://drive.google.com/file/d/13OUcV3vBNvmYY3AdG6auKyZPMSSbERvG/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1gq_5lOduqEHh7bVeOTGuJIv1hxhupihD/view?usp=drive_link)
RetinaNet (Swin)|0.311|0.451|Object detection|[config](https://drive.google.com/file/d/1I0JFGF1c41CfsfL3iJhsxzaXRbqLJGQB/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1aGCy6I-V3Flhnbsr4BDWPICkgThKSE-D/view?usp=drive_link)
Mask R-CNN (Swin)|0.364|0.534|Object detection & Instance segmentation|[config](https://drive.google.com/file/d/15TyGcLEkcV5LYeCPlFXXUPRh3MzFBXKy/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1oz_-0qJEonqwdtuahQoSECWMXmTxTb1u/view?usp=drive_link)
PVT v2|0.419|0.558|Object detection|[config](https://drive.google.com/file/d/18J7za6ood9XZrhqdCP_vhD6506sYXP60/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1cCBQdRH1McPh7aGmRvx9zkUMR-qGOzfN/view?usp=drive_link)

### Year built
Method | $mAP$ | $mAP_{50}$ | Task | config | Download
-------|-------|-------|---------|-----|--------|
RetinaNet|0.232|0.322|Object detection|[config](https://drive.google.com/file/d/10BH2DF7A7eulf6WEoYgcEWLhwBrH0ukJ/view?usp=drive_link)|[Model](https://drive.google.com/file/d/17duH60XmB5Avy-Q3eH_tMfRIT-iCzRvo/view?usp=drive_link)
YOLO-F|0.228|0.381|Object detection|[config](https://drive.google.com/file/d/1klTvgHDOoCzyJ6nIMt7ZoO-gYbBnQv5n/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1_JXVw1kAormEvAiZA0wsjNsXinGYkU43/view?usp=drive_link)
TOOD|0.180|0.241|Object detection|[config](https://drive.google.com/file/d/1OkvksiNSIzR1-DxB5gFCCxVZVeLNVngI/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1bn1bTcM3aY5kTa41OOAfikSI4KSY5uJV/view?usp=drive_link)
Faster R-CNN|0.254|0.372|Object detection|[config](https://drive.google.com/file/d/12B-3xSNfWKbYSjUwZ1gb5F-ms0Eedwzx/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1Y-TKc15Anv0NblK4ZODsgSsE2hse5Pci/view?usp=drive_link)
Mask R-CNN|0.256|0.363|Object detection & Instance segmentation|[config](https://drive.google.com/file/d/1XMcnlOrUo8kgTUnKVOesNU-5WrVVWji-/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1O4Bjtyxkxq4sSkaaGBit8CAkWqpTq41V/view?usp=drive_link)
Cascade R-CNN|0.245|0.342|Object detectionn|[config](https://drive.google.com/file/d/14QHQeqEx3sY-e3vaH72-pPcuFcYT2Whk/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1WHojQ8s09Fv5qjIkBXegZC3de8wQ1KfV/view?usp=drive_link)
RetinaNet (Swin)|0.201|0.293|Object detection|[config](https://drive.google.com/file/d/1s0fnQuHdiMFhADBPQ4tP5lFnTSEOtitW/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1PT8zEtWg6SZdJdzjnzTPABp-yyfKsiwX/view?usp=drive_link)
Mask R-CNN (Swin)|0.253|0.353|Object detection & Instance segmentation|[config](https://drive.google.com/file/d/134v1xpf785uPdCL7aM45aCMNY0M0TF9O/view?usp=drive_link)|[Model](https://drive.google.com/file/d/12LGTuZw-0QICQ1KrIZbjZYsHT81oQ9Sd/view?usp=drive_link)
PVT v2|0.258|0.359|Object detection|[config](https://drive.google.com/file/d/1_HPrHkGrXcRjdxdR9XYlnvjMhgq98Vog/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1A8FRkkDRqW3SaVCF-Hh-r3j5yXw3sbkj/view?usp=drive_link)

### Floor area ratio
Method | $mAP$ | $mAP_{50}$ | Task | config | Download
-------|-------|-------|---------|-----|--------|
RetinaNet|0.278|0.384|Object detection|[config](https://drive.google.com/file/d/1382VDyEQtaXMAkKBxaK6RkoMtI9eTA_g/view?usp=drive_link)|[Model](https://drive.google.com/file/d/11hALqGUfd4ierXDh-lhsjDwC3phBaaPX/view?usp=drive_link)
YOLO-F|0.248|0.384|Object detection|[config](https://drive.google.com/file/d/1TALAIUDh3l8CqKCdXUd4vCEIUW-ZlBnI/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1QqVCMmTK_BEayF4vgAYFPTSDL0jmrPiy/view?usp=drive_link)
TOOD|0.275|0.365|Object detection|[config](https://drive.google.com/file/d/1pCr-myStZ7y7nuYX57mSPhwaMgqvGa_v/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1uJQxKIkfOklsQ1MGeHH-NduMN0U3A7IB/view?usp=drive_link)
Faster R-CNN|0.270|0.381|Object detection|[config](https://drive.google.com/file/d/1HfcMWkzT2QUala7a2WGCXH2peEOpCZyi/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1whJt1gbsWwn3MPyqHVvGEdH9hkyh_cQm/view?usp=drive_link)
Mask R-CNN|0.279|0.285|Object detection & Instance segmentation|[config](https://drive.google.com/file/d/1-K6VN37bDbYfyNIiIxkvlpRqdrxMcflr/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1HSAaE9tAs1KpbjRkZ4dtq29g1PV6DBgp/view?usp=drive_link)
Cascade R-CNN|0.267|0.362|Object detectionn|[config](https://drive.google.com/file/d/1tDmEX0eQ0deglnB_3oFza1eH-R9VZtPW/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1_F28C5TDDBPFzNxMXhWMUpX-d5Kg9vY9/view?usp=drive_link)
RetinaNet (Swin)|0.275|0.386|Object detection|[config](https://drive.google.com/file/d/10mfZop6SNEecva74nxyGd8df35SNOWzF/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1DuAbE2tRStqVcObhmbAw_LORMl1V_Ez6/view?usp=drive_link)
Mask R-CNN (Swin)|0.269|0.367|Object detection & Instance segmentation|[config](https://drive.google.com/file/d/1nwoHGrG1mLpfTwYvWGyrtdV5CHmJBLAU/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1xfaP1sDgzZx97VDSRbMfU4heEgACOeP3/view?usp=drive_link)
PVT v2|0.269|0.365|Object detection|[config](https://drive.google.com/file/d/1S6ZZntcLACHhaOtocGg_7fDAZ3F79d1-/view?usp=drive_link)|[Model](https://drive.google.com/file/d/1DVEmmTCLfdLk1H0qAdwoDKPtFyBIQg_K/view?usp=drive_link)


## 4. Usage
### Data preparation
If you want to use your own dataset to test the models above, please prepare data following [MMdetection](https://github.com/open-mmlab/mmdetection)(Dataset in COCO format is preferred). And the data structure should be arranged as below:
```
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── Pano_Train
│   │   ├── Pano_Val
```
<!-- │   │   ├── test2017 -->
### Model test
With `OmniCity dataset`:
```
# single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}] \
    [--show]

# CPU: disable GPUs and run single-gpu testing script
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}] \
    [--show]

# multi-gpu testing
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}]
```
<!-- With `new dataset`:

* Prepare the dataset following the above rules
* Refer to the preceding operations -->

## 5. Citation
```
coming soon
```