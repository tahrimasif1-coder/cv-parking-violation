# Annotation Guidelines & SOP
**AIRI Team PITB AI Internship Task 1 — Smart Parking Violation Detection**

This document details the Standard Operating Procedure (SOP) for manually annotating and verifying image samples for YOLOv8 object detection.

---

## 1. YOLO Annotation Format Specifications

Every image `image_name.jpg` must have a corresponding text file `image_name.txt` stored under `dataset/labels/{train,val,test}/`.

### Syntax Structure:
```text
<class_id> <x_center> <y_center> <width> <height>
```

### Parameters:
- `<class_id>`: Integer index matching dataset class definitions (`0` to `3`).
- `<x_center>`: Horizontal center coordinate of the bounding box, normalized by image width (`0.0` to `1.0`).
- `<y_center>`: Vertical center coordinate of the bounding box, normalized by image height (`0.0` to `1.0`).
- `<width>`: Total bounding box width, normalized by image width (`0.0` to `1.0`).
- `<height>`: Total bounding box height, normalized by image height (`0.0` to `1.0`).

---

## 2. Definitive Class Definitions & Annotation Rules

| Class ID | Class Name | Target Object & Bounding Box Representation |
|---|---|---|
| **0** | `car` | Standard passenger vehicles, sedans, SUVs, trucks moving or parked in regular traffic lanes. Draw tight bounding box enclosing entire visible vehicle body. |
| **1** | `motorcycle` | Motorcycles, motorbikes, scooters, mopeds. Include helmet/handlebars if attached to parked vehicle. |
| **2** | `no_parking_sign` | Official circular or rectangular red & blue 'No Parking' / 'No Stopping' signs. Enclose sign face tightly; exclude supporting tall posts where possible. |
| **3** | `roadside_parking` | **Roadside Parked Vehicles / Curb Parking Instances**. Draw tight bounding box around any vehicle (car, van, truck) specifically parked along the roadside curb or in a designated/prohibited roadside parking bay. |

---

## 3. Step-by-Step Manual Annotation Guide

### Option A: Using Roboflow (Recommended Web Tool)
1. Sign in to [Roboflow](https://roboflow.com/) and create a project named `smart-parking-violation`.
2. Upload raw images (`dataset/images/raw/`).
3. Set classes: `car`, `motorcycle`, `no_parking_sign`, `roadside_parking`.
4. Use the rectangle tool to draw tight bounding boxes around objects according to the rules above.
5. Export dataset selecting **YOLOv8** format.

### Option B: Using CVAT (Computer Vision Annotation Tool)
1. Launch CVAT online or locally via Docker.
2. Create task with the 4 class labels: `car`, `motorcycle`, `no_parking_sign`, `roadside_parking`.
3. Draw bounding boxes ensuring no loose borders around objects.
4. Export task as `YOLO 1.1` format.

---

## 4. Quality Assurance (QA) Checklist

- [x] **No Loose Margins**: Ensure boxes tightly bound target objects without excessive background.
- [x] **Occlusion Handling**: If an object is partially occluded (>50% visible), draw the box over the visible portion.
- [x] **Truncated Objects**: For vehicles partially outside frame, draw box up to image edge.
- [x] **Class Index Verification**: Ensure class index matches `data.yaml` integer indices (`0` to `3`) exactly.
