import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:face_detection_flutter/BoundingBox.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as img;

import 'Model/AnchorOption.dart';
import 'Model/Detection.dart';
import 'Model/OptionsFace.dart';
import 'UtilsFace.dart';

void main() {
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false,
    home: MyApp(),
  ));
}

class MyApp extends StatefulWidget {
  @override
  State<StatefulWidget> createState() {
    return MyAppState();
  }
}

class MyAppState extends State<MyApp> {
  CameraDescription camera;
  CameraController controller;
  Future<void> _initializeControllerFuture;

  List<Anchor> _anchors = new List();
  ImageProcessor _imageProcessor;
  Interpreter _interpreter;
  OptionsFace options;
  AnchorOption anchors;
  List<Detection> _detections;

  bool _isDetecting = false;
  Directory dir;

  @override
  void initState() {
    _detections = List<Detection>();
    _loadModel();
  }

  @override
  void dispose() {
    controller.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return controller == null
        ? Center(
            child: CircularProgressIndicator(),
          )
        : Stack(
          children: [
            Container(
              child: AspectRatio(
                aspectRatio: controller.value.aspectRatio,
                child: CameraPreview(controller),
              ),
            ),

            BoundingBox(_detections, 640, 480, MediaQuery.of(context).size.height, MediaQuery.of(context).size.width),
          ],
    );
  }

  void _loadModel() async {
    dir = await getApplicationDocumentsDirectory();
    options = OptionsFace(
        numClasses: 1,
        numBoxes: 896,
        numCoords: 16,
        keypointCoordOffset: 4,
        ignoreClasses: [],
        scoreClippingThresh: 100.0,
        minScoreThresh: 0.75,
        numKeypoints: 6,
        numValuesPerKeypoint: 2,
        reverseOutputOrder: false,
        boxCoordOffset: 0,
        xScale: 128,
        yScale: 128,
        hScale: 128,
        wScale: 128);

    anchors = AnchorOption(
        inputSizeHeight: 128,
        inputSizeWidth: 128,
        minScale: 0.1484375,
        maxScale: 0.75,
        anchorOffsetX: 0.5,
        anchorOffsetY: 0.5,
        numLayers: 4,
        featureMapHeight: [],
        featureMapWidth: [],
        strides: [8, 16, 16, 16],
        aspectRatios: [1.0],
        reduceBoxesInLowestLayer: false,
        interpolatedScaleAspectRatio: 1.0,
        fixedAnchorSize: true);

    NormalizeOp _normalizeInput = NormalizeOp(127.5, 127.5);

    _anchors = UtilsFace().getAnchors(anchors);
    _interpreter = await Interpreter.fromAsset("face_detection_front.tflite");
    var _inputShape = _interpreter.getInputTensor(0).shape;
    _imageProcessor = ImageProcessorBuilder()
        .add(ResizeOp(
            _inputShape[1], _inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR))
        .add(_normalizeInput)
        .build();
    await _onStream();
  }

  _onStream() async {
    WidgetsFlutterBinding.ensureInitialized();
    camera = (await availableCameras())[1];

    controller = CameraController(camera, ResolutionPreset.medium);
    await controller.initialize();

    controller = CameraController(camera, ResolutionPreset.medium);
    await controller.initialize();
    setState(() {});
    await controller.startImageStream((CameraImage image) async {
      if (_isDetecting) return;
      _isDetecting = true;
      await Future.delayed(const Duration(seconds: 1), () {
        _tfLite(image);
        _isDetecting = false;
      });
      print("---- ${DateTime.now()} ----\n");
    });
  }

  void _tfLite(CameraImage image) async {
    img.Image _img;
    _img = img.Image.fromBytes(
        image.width, image.height, _concatenatePlanes(image.planes));

    TensorImage tensorImage = TensorImage.fromImage(_img);
    tensorImage = _imageProcessor.process(tensorImage);

    TensorBuffer output0 = TensorBuffer.createFixedSize(
        _interpreter.getOutputTensor(0).shape,
        _interpreter.getOutputTensor(0).type);
    TensorBuffer output1 = TensorBuffer.createFixedSize(
        _interpreter.getOutputTensor(1).shape,
        _interpreter.getOutputTensor(1).type);

    Map<int, ByteBuffer> outputs = {0: output0.buffer, 1: output1.buffer};

    _interpreter.runForMultipleInputs([tensorImage.buffer], outputs);

    List<double> regression = output0.getDoubleList();
    List<double> classificators = output1.getDoubleList();

    List<Detection> detections = UtilsFace().process(
        options: options,
        rawScores: classificators,
        rawBoxes: regression,
        anchors: _anchors);

    _detections = UtilsFace().origNms(detections, 0.75);

    for (Detection detection in _detections) {
      print(detection.classID.toString() +
          " " +
          detection.xMin.toString() +
          " " +
          detection.yMin.toString() +
          " " +
          detection.width.toString() +
          " " +
          detection.height.toString() +
          " " +
          detection.score.toString());

      var face = img.copyCrop(
          _img,
          _img.width * detection.xMin.toInt(),
          _img.height * detection.yMin.toInt(),
          _img.width * detection.width.toInt(),
          _img.height * detection.height.toInt());
    }

    setState(() {});
  }

  Uint8List _concatenatePlanes(List<Plane> planes) {
    final WriteBuffer allBytes = WriteBuffer();
    for (Plane plane in planes) {
      allBytes.putUint8List(plane.bytes);
    }
    return allBytes.done().buffer.asUint8List();
  }
}
