using Emgu.CV;
using Emgu.CV.CvEnum;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SentryDeen.EdgeWorker;

public class Worker : BackgroundService
{
    private readonly ILogger<Worker> _logger;
    private readonly string _streamSource = "0"; 
    private readonly int _targetFps = 10;
    
    // AI Infrastructure Configuration
    private readonly string _modelPath = "yolov10n.onnx";
    private InferenceSession? _onnxSession;

    public Worker(ILogger<Worker> logger)
    {
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Initializing SentryDeen AI Model: {Model}...", _modelPath);
        
        // Boot up the ONNX Execution Engine Session
        _onnxSession = new InferenceSession(_modelPath);

        _logger.LogInformation("ONNX AI Session initialized successfully. Starting video capture...");

        using var capture = int.TryParse(_streamSource, out int camIndex) 
            ? new VideoCapture(camIndex) 
            : new VideoCapture(_streamSource);

        if (!capture.IsOpened)
        {
            _logger.LogCritical("CRITICAL: Failed to connect to stream source: {Source}", _streamSource);
            return;
        }

        using var rawFrame = new Mat();
        int executionDelayMs = 1000 / _targetFps;

        while (!stoppingToken.IsCancellationRequested)
        {
            var loopStartTime = DateTime.UtcNow;

            if (capture.Grab())
            {
                capture.Retrieve(rawFrame);

                if (!rawFrame.IsEmpty)
                {
                    // Execute AI Inference on the current frame
                    RunAIInference(rawFrame);
                }
            }

            var loopElapsedTime = (int)(DateTime.UtcNow - loopStartTime).TotalMilliseconds;
            int remainingDelay = Math.Max(0, executionDelayMs - loopElapsedTime);

            await Task.Delay(remainingDelay, stoppingToken);
        }
    }

    private void RunAIInference(Mat frame)
    {
        // 1. Resize input image to the exact dimensions the AI model expects (640x640)
        using var resizedFrame = new Mat();
        CvInvoke.Resize(frame, resizedFrame, new System.Drawing.Size(640, 640), 0, 0, Inter.Linear);

        // 2. Transform the raw image pixels into an optimized 4D mathematical tensor
        var tensor = ConvertMatToTensor(resizedFrame);

        // 3. Prepare the input metadata container for ONNX Runtime
        var inputName = _onnxSession!.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

        // 4. Fire the inference engine directly through the model network path
        using var results = _onnxSession.Run(inputs);
        
        // 5. Extract the raw numeric data array out of the ONNX results wrapper
        var output = results.First().AsTensor<float>();
        
        // YOLOv10 outputs up to 300 possible guess arrays per frame
        // Each guess contains 6 pieces of structural data
        int numDetections = output.Dimensions[1]; 

        for (int i = 0; i < numDetections; i++)
        {
            // Extract the confidence score (Index 4)
            float confidence = output[0, i, 4];

            // FILTER: If the AI is less than 50% sure, ignore it to prevent ghost alerts
            if (confidence > 0.50f)
            {
                // Extract the Class ID (Index 5)
                int classId = (int)output[0, i, 5];

                // In the standard COCO dataset, Class ID 0 is a Human Person
                if (classId == 0)
                {
                    _logger.LogWarning("ALERT -> Detected: PERSON | Confidence: {Confidence}%", 
                        Math.Round(confidence * 100, 1));
                }
                else
                {
                    // Log other objects it detects with decent confidence
                    _logger.LogInformation("Detected Object Class ID: {ClassId} | Confidence: {Confidence}%", 
                        classId, Math.Round(confidence * 100, 1));
                }
            }
        }
    }

    private DenseTensor<float> ConvertMatToTensor(Mat image)
    {
        // Allocate a 4D float array tracking: [BatchSize (1), ColorChannels (3), Height (640), Width (640)]
        var tensor = new DenseTensor<float>(new[] { 1, 3, 640, 640 });
        
        // Extract raw pixel data out of OpenCV memory paths
        var imageSize = image.Width * image.Height;
        byte[] rawBytes = new byte[imageSize * 3];
        image.CopyTo(rawBytes);

        // Normalize color dimensions cleanly from bytes (0-255) to float decimals (0.0 - 1.0)
        for (int y = 0; y < 640; y++)
        {
            for (int x = 0; x < 640; x++)
            {
                int index = (y * 640 + x) * 3;
                
                // OpenCV maps image colors as Blue, Green, Red (BGR). We map them out to tensor cells:
                tensor[0, 0, y, x] = rawBytes[index + 2] / 255.0f; // Red Channel
                tensor[0, 1, y, x] = rawBytes[index + 1] / 255.0f; // Green Channel
                tensor[0, 2, y, x] = rawBytes[index + 0] / 255.0f; // Blue Channel
            }
        }

        return tensor;
    }
}