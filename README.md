# pose-estimation
Tests/demos of various pre-trained pose estimation models.

[Singlepose Example](https://youtu.be/N1KlFnjoEtg), [Multipose Example](https://youtu.be/XxJNebC_oqc)

## Setup

After cloning, create a new folder called ``data`` in the root of the repository. Then create another folder inside ``data`` called ``input``.
Finally, place the video you wish to perform pose estimation on inside of ``data/input``, and rename it to ``pose_estimation_benchmark2.mp4``.
Once this is done, you can run ``movenet.py`` and the output will be saved to ``data/output``.

## Docker Environment
Building image:
```
docker-compose build
```

Running environment:
```
docker-compose run --rm app
```

## Example Output
<img src="./images/demo_image.jpg" width="270px"></img>
