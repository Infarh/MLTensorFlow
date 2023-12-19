using Microsoft.ML;
using Microsoft.ML.Data;

var _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
var images_folder = Path.Combine(_assetsPath, "images");
var _trainTagsTsv = Path.Combine(images_folder, "tags.tsv");
var _testTagsTsv = Path.Combine(images_folder, "test-tags.tsv");
var _predictSingleImage = Path.Combine(images_folder, "toaster3.jpg");
var inception_tensor_flow_model = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

var ml = new MLContext();

var model = GenerateModel(ml);

ClassifySingleImage(ml, model);

Console.WriteLine("End.");

return;

static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
{
    foreach (var prediction in imagePredictionData)
    {
        Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
    }
}

void ClassifySingleImage(MLContext mlContext, ITransformer model)
{
    var imageData = new ImageData()
    {
        ImagePath = _predictSingleImage
    };

    // Make prediction function (input = ImageData, output = ImagePrediction)
    var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
    var prediction = predictor.Predict(imageData);

    Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
}

ITransformer GenerateModel(MLContext ml)
{

    var pipeline = ml.Transforms
        .LoadImages(outputColumnName: "input", imageFolder: images_folder, inputColumnName: nameof(ImageData.ImagePath))
        .Append(ml.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
        .Append(ml.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
        .Append(ml.Model.LoadTensorFlowModel(inception_tensor_flow_model).ScoreTensorFlowModel(outputColumnNames: ["softmax2_pre_activation"], inputColumnNames: ["input"], addBatchDimensionInput: true))
        .Append(ml.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
        .Append(ml.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
        .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
        .AppendCacheCheckpoint(ml);

    var trainingData = ml.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);

    var model = pipeline.Fit(trainingData);

    var testData = ml.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
    var predictions = model.Transform(testData);

    // Create an IEnumerable for the predictions for displaying results
    var imagePredictionData = ml.Data.CreateEnumerable<ImagePrediction>(predictions, true);
    DisplayResults(imagePredictionData);

    var metrics = ml.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey", predictedLabelColumnName: "PredictedLabel");

    Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
    Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

    return model;
}

public class ImageData
{
    [LoadColumn(0)]
    public string? ImagePath;

    [LoadColumn(1)]
    public string? Label;
}

public class ImagePrediction : ImageData
{
    public float[]? Score;

    public string? PredictedLabelValue;
}

struct InceptionSettings
{
    public const int ImageHeight = 224;
    public const int ImageWidth = 224;
    public const float Mean = 117;
    public const float Scale = 1;
    public const bool ChannelsLast = true;
}