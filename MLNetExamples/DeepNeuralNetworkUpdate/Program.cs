using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System;
using System.IO;
using System.Linq;

namespace DeepNeuralNetworkUpdate
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

            var options = new ImageClassificationTrainer.Options()
            {
                Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                Epoch = 100,
                BatchSize = 20,
                LearningRate = 0.01f,
                LabelColumnName = "LabelKey",
                FeatureColumnName = "Images",
                ValidationSet = validationData
            };

            var pipeline = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Transforms.LoadRawImageBytes("Images", imagesFolder, "ImagePath"))
                .Append(context.MulticlassClassification.Trainers.ImageClassification(options));

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

            context.Model.Save(model, imageData.Schema, "./dnn_model.zip");

        }
    }
}
