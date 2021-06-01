import 'package:scidart/numdart.dart';

import 'Model/AnchorOption.dart';
import 'Model/Detection.dart';
import 'Model/OptionsFace.dart';

class UtilsFace {
  List<Anchor> getAnchors(AnchorOption options) {
    List<Anchor> _anchors = [];
    if (options.stridesSize != options.numLayers) {
      print('strides_size and num_layers must be equal.');
      return [];
    }
    int layerID = 0;
    while (layerID < options.stridesSize) {
      List<double> anchorHeight = [];
      List<double> anchorWidth = [];
      List<double?> aspectRatios = [];
      List<double> scales = [];

      int lastSameStrideLayer = layerID;
      while (lastSameStrideLayer < options.stridesSize &&
          options.strides![lastSameStrideLayer] == options.strides![layerID]) {
        double scale = options.minScale! +
            (options.maxScale! - options.minScale!) *
                1.0 *
                lastSameStrideLayer /
                (options.stridesSize - 1.0);
        if (lastSameStrideLayer == 0 && options.reduceBoxesInLowestLayer!) {
          aspectRatios.add(1.0);
          aspectRatios.add(2.0);
          aspectRatios.add(0.5);
          scales.add(0.1);
          scales.add(scale);
          scales.add(scale);
        } else {
          for (int i = 0; i < options.aspectRatios!.length; i++) {
            aspectRatios.add(options.aspectRatios![i]);
            scales.add(scale);
          }

          if (options.interpolatedScaleAspectRatio! > 0.0) {
            double scaleNext = 0.0;
            if (lastSameStrideLayer == options.stridesSize - 1) {
              scaleNext = 1.0;
            } else {
              scaleNext = options.minScale! +
                  (options.maxScale! - options.minScale!) *
                      1.0 *
                      (lastSameStrideLayer + 1) /
                      (options.stridesSize - 1.0);
            }
            scales.add(sqrt(scale * scaleNext));
            aspectRatios.add(options.interpolatedScaleAspectRatio);
          }
        }
        lastSameStrideLayer++;
      }
      for (int i = 0; i < aspectRatios.length; i++) {
        double ratioSQRT = sqrt(aspectRatios[i]!);
        anchorHeight.add(scales[i] / ratioSQRT);
        anchorWidth.add(scales[i] * ratioSQRT);
      }
      int featureMapHeight = 0;
      int featureMapWidth = 0;
      if (options.featureMapHeightSize > 0) {
        featureMapHeight = options.featureMapHeight![layerID];
        featureMapWidth = options.featureMapWidth![layerID];
      } else {
        int stride = options.strides![layerID];
        featureMapHeight = (1.0 * options.inputSizeHeight! / stride).ceil();
        featureMapWidth = (1.0 * options.inputSizeWidth! / stride).ceil();
      }

      for (int y = 0; y < featureMapHeight; y++) {
        for (int x = 0; x < featureMapWidth; x++) {
          for (int anchorID = 0; anchorID < anchorHeight.length; anchorID++) {
            double xCenter =
                (x + options.anchorOffsetX!) * 1.0 / featureMapWidth;
            double yCenter =
                (y + options.anchorOffsetY!) * 1.0 / featureMapHeight;
            double w = 0;
            double h = 0;
            if (options.fixedAnchorSize!) {
              w = 1.0;
              h = 1.0;
            } else {
              w = anchorWidth[anchorID];
              h = anchorHeight[anchorID];
            }
            _anchors.add(Anchor(xCenter, yCenter, h, w));
          }
        }
      }
      layerID = lastSameStrideLayer;
    }
    return _anchors;
  }

  List<Detection> process(
      {required OptionsFace options,
      List<double>? rawScores,
      List<double>? rawBoxes,
      List<Anchor>? anchors}) {
    List<double> detectionScores = [];
    List<int> detectionClasses = [];

    int boxes = options.numBoxes!;
    for (int i = 0; i < boxes; i++) {
      int classId = -1;
      double maxScore = double.minPositive;
      for (int scoreIdx = 0; scoreIdx < options.numClasses!; scoreIdx++) {
        double? score = rawScores![i * options.numClasses! + scoreIdx];
        if (options.sigmoidScore) {
          if (options.scoreClippingThresh! > 0) {
            if (score < -options.scoreClippingThresh!)
              score = -options.scoreClippingThresh!;
            if (score > options.scoreClippingThresh!)
              score = options.scoreClippingThresh;
            score = 1.0 / (1.0 + exp(-score!));
            if (maxScore < score) {
              maxScore = score;
              classId = scoreIdx;
            }
          }
        }
      }
      detectionClasses.add(classId);
      detectionScores.add(maxScore);
    }
    List<Detection> detections = convertToDetections(
        rawBoxes, anchors, detectionScores, detectionClasses, options);
    return detections;
  }

  List<Detection> convertToDetections(
      List<double>? rawBoxes,
      List<Anchor>? anchors,
      List<double> detectionScores,
      List<int> detectionClasses,
      OptionsFace options) {
    List<Detection> _outputDetections = [];
    for (int i = 0; i < options.numBoxes!; i++) {
      if (detectionScores[i] < options.minScoreThresh!) continue;
      int boxOffset = 0;
      Array boxData = decodeBox(rawBoxes!, i, anchors!, options);
      Detection detection = convertToDetection(
          boxData[boxOffset + 0],
          boxData[boxOffset + 1],
          boxData[boxOffset + 2],
          boxData[boxOffset + 3],
          detectionScores[i],
          detectionClasses[i],
          options.flipVertically);
      _outputDetections.add(detection);
    }
    return _outputDetections;
  }

  Array decodeBox(
      List<double> rawBoxes, int i, List<Anchor> anchors, OptionsFace options) {
    Array boxData =
        Array(List<double>.generate(options.numCoords!, (i) => 0.0));
    int boxOffset = i * options.numCoords! + options.boxCoordOffset;
    double yCenter = rawBoxes[boxOffset];
    double xCenter = rawBoxes[boxOffset + 1];
    double h = rawBoxes[boxOffset + 2];
    double w = rawBoxes[boxOffset + 3];
    if (options.reverseOutputOrder) {
      xCenter = rawBoxes[boxOffset];
      yCenter = rawBoxes[boxOffset + 1];
      w = rawBoxes[boxOffset + 2];
      h = rawBoxes[boxOffset + 3];
    }

    xCenter = xCenter / options.xScale * anchors[i].w + anchors[i].xCenter;
    yCenter = yCenter / options.yScale * anchors[i].h + anchors[i].yCenter;

    if (options.applyExponentialOnBoxSize) {
      h = exp(h / options.hScale) * anchors[i].h;
      w = exp(w / options.wScale) * anchors[i].w;
    } else {
      h = h / options.hScale * anchors[i].h;
      w = w / options.wScale * anchors[i].w;
    }

    double yMin = yCenter - h / 2.0;
    double xMin = xCenter - w / 2.0;
    double yMax = yCenter + h / 2.0;
    double xMax = xCenter + w / 2.0;

    boxData[0] = yMin;
    boxData[1] = xMin;
    boxData[2] = yMax;
    boxData[3] = xMax;

    if (options.numKeypoints > 0) {
      for (int k = 0; k < options.numKeypoints; k++) {
        int offset = i * options.numCoords! +
            options.keypointCoordOffset! +
            k * options.numValuesPerKeypoint;
        double keyPointY = rawBoxes[offset];
        double keyPointX = rawBoxes[offset + 1];

        if (options.reverseOutputOrder) {
          keyPointX = rawBoxes[offset];
          keyPointY = rawBoxes[offset + 1];
        }
        boxData[4 + k * options.numValuesPerKeypoint] =
            keyPointX / options.xScale * anchors[i].w + anchors[i].xCenter;

        boxData[4 + k * options.numValuesPerKeypoint + 1] =
            keyPointY / options.yScale * anchors[i].h + anchors[i].yCenter;
      }
    }
    return boxData;
  }

  Detection convertToDetection(double boxYMin, double boxXMin, double boxYMax,
      double boxXMax, double score, int classID, bool flipVertically) {
    double _yMin;
    if (flipVertically)
      _yMin = 1.0 - boxYMax;
    else
      _yMin = boxYMin;
    return new Detection(score, classID, boxXMin, _yMin, (boxXMax - boxXMin),
        (boxXMax - boxYMin));
  }

  List<Detection> origNms(List<Detection> detections, double threshold) {
    if (detections.length <= 0) return [];
    List<double> x1 = [];
    List<double> x2 = [];
    List<double> y1 = [];
    List<double> y2 = [];
    List<double> s = [];

    detections.forEach((detection) {
      x1.add(detection.xMin);
      x2.add(detection.xMin + detection.width);
      y1.add(detection.yMin);
      y2.add(detection.yMin + detection.height);
      s.add(detection.score);
    });

    Array _x1 = new Array(x1);
    Array _x2 = new Array(x2);
    Array _y1 = new Array(y1);
    Array _y2 = new Array(y2);

    Array area = (_x2 - _x1) * (_y2 - _y1);
    List<double> I = _quickSort(s);
    List<int> positions = [];
    I.forEach((element) {
      positions.add(s.indexOf(element));
    });

    List<int> ind0 = positions.sublist(positions.length - 1, positions.length);
    List<int> ind1 = positions.sublist(0, positions.length - 1);

    List<int> pick = [];
    while (I.length > 0) {
      Array xx1 = _maximum(_itemIndex(_x1, ind0)[0], _itemIndex(_x1, ind1));
      Array yy1 = _maximum(_itemIndex(_y1, ind0)[0], _itemIndex(_y1, ind1));
      Array xx2 = _maximum(_itemIndex(_x2, ind0)[0], _itemIndex(_x2, ind1));
      Array yy2 = _maximum(_itemIndex(_y2, ind0)[0], _itemIndex(_y2, ind1));
      Array w = _maximum(0.0, xx2 - xx1);
      Array h = _maximum(0.0, yy2 - yy1);
      Array inter = w * h;
      Array o = inter /
          (_sum(_itemIndex(area, ind0)[0], _itemIndex(area, ind1)) - inter);
      pick.add(ind0[0]);
      I = o.where((element) => element <= threshold).toList();
    }
    return [detections[pick[0]]];
  }

  Array _sum(double a, Array b) {
    List<double> _temp = [];
    b.forEach((element) {
      _temp.add(a + element);
    });
    return new Array(_temp);
  }

  Array _maximum(double value, Array itemIndex) {
    List<double> _temp = [];
    itemIndex.forEach((element) {
      if (value > element)
        _temp.add(value);
      else
        _temp.add(element);
    });
    return new Array(_temp);
  }

  Array _itemIndex(Array item, List<int> positions) {
    List<double> _temp = [];
    positions.forEach((element) => _temp.add(item[element]));
    return new Array(_temp);
  }

  List<double> _quickSort(List<double> a) {
    if (a.length <= 1) return a;

    var pivot = a[0];
    List<double> less = [];
    List<double> more = [];
    List<double> pivotList = [];

    a.forEach((var i) {
      if (i.compareTo(pivot) < 0) {
        less.add(i);
      } else if (i.compareTo(pivot) > 0) {
        more.add(i);
      } else {
        pivotList.add(i);
      }
    });

    less = _quickSort(less);
    more = _quickSort(more);

    less.addAll(pivotList);
    less.addAll(more);
    return less;
  }
}
