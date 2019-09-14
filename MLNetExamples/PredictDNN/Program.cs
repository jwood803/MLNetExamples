using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.WindowsAzure.Storage;
using Microsoft.WindowsAzure.Storage.Blob;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace PredictDNN
{
    class Program
    {
        static string connectionString = "";

        static async Task Main(string[] args)
        {
            var context = new MLContext();

            var testImagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "images");

            var testFiles = Directory.GetFiles(testImagesFolder, "*", SearchOption.AllDirectories);

            var testImages = testFiles.Select(file => new ImageData
            {
                ImagePath = file
            });

            var model = context.Model.Load("./dnn_model.zip", out var inputSchema);

            var predictionEngine = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

            VBuffer<ReadOnlyMemory<char>> keys = default;
            predictionEngine.OutputSchema["LabelKey"].GetKeyValues(ref keys);

            var originalLabels = keys.DenseValues().ToArray();

            Console.WriteLine(Environment.NewLine);

            foreach (var image in testImages)
            {
                var prediction = predictionEngine.Predict(image);

                Console.WriteLine($"Image : {Path.GetFileName(image.ImagePath)}, Score : {prediction.Score.Max()}, Predicted Label : {originalLabels[prediction.PredictedLabel]}");
            }

            var storageAccount = CloudStorageAccount.Parse(connectionString);

            var client = storageAccount.CreateCloudBlobClient();

            var container = client.GetContainerReference("images");

            var images = await container.ListBlobsSegmentedAsync(null);

            foreach (CloudBlockBlob image in images.Results)
            {
                var blob = container.GetBlockBlobReference(image.Name);

                await blob.DownloadToFileAsync($"./{image.Name}", FileMode.Create);

                var newImage = new ImageData
                {
                    ImagePath = $"./{image.Name}"
                };

                var prediction = predictionEngine.Predict(newImage);

                Console.WriteLine($"Image : {image.Name}, Score : {prediction.Score.Max()}, Predicted Label : {originalLabels[prediction.PredictedLabel]}");
            }

            Console.ReadLine();
        }
    }
}
