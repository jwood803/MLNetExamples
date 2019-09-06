using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

namespace DeepNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var imagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "images");

            var files = Directory.GetFiles(imagesFolder, "*", SearchOption.AllDirectories);

            var images = files.Select(file => new ImageData
                {
                    ImagePath = file,
                    Label = Directory.GetParent(file).Name
                });

            var context = new MLContext();

            var imageData = context.Data.LoadFromEnumerable(images);
            var imageDataShuffled = context.Data.ShuffleRows(imageData);

            var testTrainData = context.Data.TrainTestSplit(imageDataShuffled, testFraction: 0.2);

            var validationData = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Fit(testTrainData.TestSet)
                .Transform(testTrainData.TestSet);

            var pipeline = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Model.ImageClassification(
                    "ImagePath", 
                    "LabelKey", 
                    arch: Microsoft.ML.Transforms.ImageClassificationEstimator.Architecture.ResnetV2101,
                    epoch: 100,
                    batchSize: 10,
                    metricsCallback: Console.WriteLine,
                    validationSet: validationData));

            var model = pipeline.Fit(testTrainData.TrainSet);

            var predicions = model.Transform(testTrainData.TestSet);

            var metrics = context.MulticlassClassification.Evaluate(predicions, labelColumnName: "LabelKey", predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine(Environment.NewLine);
            Console.WriteLine($"Log loss - {metrics.LogLoss}");

            var predictionEngine = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

            var testImagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "test");

            var testFiles = Directory.GetFiles(testImagesFolder, "*", SearchOption.AllDirectories);

            var testImages = testFiles.Select(file => new ImageData
            {
                ImagePath = file
            });

            VBuffer<ReadOnlyMemory<char>> keys = default;
            predictionEngine.OutputSchema["LabelKey"].GetKeyValues(ref keys);

            var originalLabels = keys.DenseValues().ToArray();

            Console.WriteLine(Environment.NewLine);

            foreach (var image in testImages)
            {
                var prediction = predictionEngine.Predict(image);

                var labelIndex = prediction.PredictedLabel;

                Console.WriteLine($"Image : {Path.GetFileName(image.ImagePath)}, Score : {prediction.Score.Max()}, Predicted Label : {originalLabels[labelIndex]}");
            }

            Console.ReadLine();
        }
    }
}
