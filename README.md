# pose-estimation
Demo of movenet pose estimation. Can be used to create videos with overlayed pose estimations for up to six people.

[Singlepose Example](https://youtu.be/N1KlFnjoEtg), [Multipose Example](https://youtu.be/XxJNebC_oqc)

## Usage

Run ``movenet.py`` with ``--input_path`` set to the input video path. By default the output will be saved to ``./data``.

Example:

```
python movenet.py --path ../data/input/pose_estimation_video.mp4
```

Full list of parameters:

- ``--input_path``: Path to input video file. Default: ``None``
- ``--output_path``: Path to output directroy. Default: ``./data``
- ``--confidence_threshold``: Confidence threshold for keypoints. Default: ``0.1``

## Docker Environment
Building image:
```
docker-compose build
```

Starting container/environment:
```
docker-compose up -d
```

Opening a shell in container:
```
docker-compose exec pose-estimation bash
```

Instead of opening a shell, you can also go to http://localhost:8888/ to access a Jupyter Lab instance running inside the container.

Stopping container/environment:
```
docker-compose down
```

## Example Output
<img src="./images/demo_image.jpg" width="270px"></img>
