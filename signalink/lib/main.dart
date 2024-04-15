import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite/tflite.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(camera: cameras[0]));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Signalink',
      theme: ThemeData(
        primarySwatch: Colors.yellow,
        primaryColor: Colors.yellow,
      ),
      home: MyHomePage(camera: camera),
    );
  }
}

class MyHomePage extends StatefulWidget {
  final CameraDescription camera;

  const MyHomePage({super.key, required this.camera});

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late CameraController _controller;
  List<dynamic> _output = [];
  bool _loading = false;
  bool _modelLoaded = false;
  bool _isFrontCamera = false;
  bool _isFlashOn = false; // Track whether the flash is turned on

  @override
  void initState() {
    super.initState();
    _loading = true;
    _loadModel();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    CameraDescription selectedCamera = widget.camera;
    if (_isFrontCamera) {
      final cameras = await availableCameras();
      selectedCamera = cameras.firstWhere((camera) => camera.lensDirection == CameraLensDirection.front);
    }

    _controller = CameraController(
      selectedCamera,
      ResolutionPreset.medium,
      enableAudio: false, // Disable audio to prevent audio errors on some devices
    );

    await _controller.initialize();

    if (mounted) {
      setState(() {});
      _controller.startImageStream((CameraImage img) {
        if (_loading) return;
        _loading = true;
        _classifyImage(img);
      });
    }
  }

  Future<void> _loadModel() async {
    await Tflite.loadModel(
      model: 'assets/model_unquant.tflite',
      labels: 'assets/labels.txt',
    );
    setState(() {
      _loading = false;
      _modelLoaded = true;
    });
  }

  @override
  void dispose() {
    Tflite.close();
    _controller.dispose();
    super.dispose();
  }

  void _classifyImage(CameraImage img) async {
    if (!_modelLoaded) return;

    var output = await Tflite.runModelOnFrame(
      bytesList: img.planes.map((plane) {
        return plane.bytes;
      }).toList(),
      imageHeight: img.height,
      imageWidth: img.width,
      imageMean: 127.5,
      imageStd: 127.5,
      rotation: 90,
      numResults: 1,
    );

    setState(() {
      _output = output!;
      _loading = false;
    });
  }

  void _flipCamera() {
    _isFrontCamera = !_isFrontCamera;
    _initializeCamera();
  }

  void _toggleFlash() {
    _isFlashOn = !_isFlashOn;
    _controller.setFlashMode(_isFlashOn ? FlashMode.torch : FlashMode.off);
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Container();
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text('Signalink'),
      ),
      body: Stack(
        children: <Widget>[
          CameraPreview(_controller),
          Positioned(
            bottom: 16.0,
            right: 16.0,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: <Widget>[
                FloatingActionButton(
                  onPressed: _flipCamera,
                  child: const Icon(Icons.flip_camera_android),
                ),
                const SizedBox(width: 16.0),
                FloatingActionButton(
                  onPressed: _toggleFlash,
                  child: _isFlashOn ? const Icon(Icons.flash_off) : const Icon(Icons.flash_on),
                ),
              ],
            ),
          ),
          Center(
            child: _loading
                ? const CircularProgressIndicator()
                : _output.isNotEmpty
                    ? Text(
                        'Predicted ASL Sign: ${_output[0]['label']}',
                        style: const TextStyle(fontSize: 20.0),
                      )
                    : const Text(
                        'No prediction yet.',
                        style: TextStyle(fontSize: 20.0),
                      ),
          ),
        ],
      ),
    );
  }
}
