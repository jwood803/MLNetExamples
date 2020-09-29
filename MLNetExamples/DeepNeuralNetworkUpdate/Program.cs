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
                .Append(context.Transforms.LoadRawImageBytes("Image", imagesFolder, "ImagePath"))
                .Fit(testTrainData.TestSet)
                .Transform(testTrainData.TestSet);

            var options = new ImageClassificationTrainer.Options()
            {
                Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                Epoch = 100,
                BatchSize = 20,
                LearningRate = 0.01f,
                LabelColumnName = "LabelKey",
                FeatureColumnName = "Image",
                ValidationSet = validationData
            };

            var imagesPipeline = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Transforms.LoadRawImageBytes("Image", imagesFolder, "ImagePath"));

            var imageDataModel = imagesPipeline.Fit(testTrainData.TrainSet);

            var imageDataView = imageDataModel.Transform(testTrainData.TrainSet);

            var pipeline = context.MulticlassClassification.Trainers.ImageClassification(options)
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(imageDataView);

            var predictionEngine = context.Model.CreatePredictionEngine<ImageModelInput, ImagePrediction>(model);

            var testImagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "test");

            var testFiles = Directory.GetFiles(testImagesFolder, "*", SearchOption.AllDirectories);

            var testImages = testFiles.Select(file => new ImageModelInput
            {
                ImagePath = file
            });

            Console.WriteLine(Environment.NewLine);

            var testImagesData = context.Data.LoadFromEnumerable(testImages);

            var testImageDataView = imagesPipeline.Fit(testImagesData).Transform(testImagesData);

            var predictions = model.Transform(testImageDataView);

            var testPredictions = context.Data.CreateEnumerable<ImagePrediction>(predictions, reuseRowObject: false);

            foreach (var prediction in testPredictions)
            {
                var labelIndex = prediction.PredictedLabel;

                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)}, Predicted Label: {prediction.PredictedLabel}");
            }

            context.Model.Save(model, imageData.Schema, "./dnn_model.zip");
        }
    }
}
