using Azure.Messaging.EventHubs;
using Azure.Messaging.EventHubs.Consumer;
using Azure.Messaging.EventHubs.Processor;
using Azure.Storage.Blobs;
using Microsoft.ML;
using System;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace EventHubPredict
{
    class Program
    {
        private const string EventHubConnectionString = "Endpoint=sb://mlnetehn.servicebus.windows.net/;SharedAccessKeyName=Manage;SharedAccessKey=6OZXDj5GE1Z+yTc1fVlzogRQL/6YlmIKnoZwx5wZJOk=;EntityPath=mlneteh";
        private const string EventHubName = "mlneteh ";
        private const string blobStorageConnectionString = "DefaultEndpointsProtocol=https;AccountName=databricksdemostorage;AccountKey=+z/ldXQTw4r4pWaEHFg8W+6Sl8AraGQzrN9w3XgRrWiM11PTOcuYjtLrrKBULVf4tTILfyklWxL8DoKilBBEYg==;BlobEndpoint=https://databricksdemostorage.blob.core.windows.net/;QueueEndpoint=https://databricksdemostorage.queue.core.windows.net/;TableEndpoint=https://databricksdemostorage.table.core.windows.net/;FileEndpoint=https://databricksdemostorage.file.core.windows.net/;";
        private const string blobContainerName = "events";

        static async Task Main(string[] args)
        {
            string consumerGroup = EventHubConsumerClient.DefaultConsumerGroupName;

            var storageClient = new BlobContainerClient(blobStorageConnectionString, blobContainerName);

            var processor = new EventProcessorClient(storageClient, consumerGroup, EventHubConnectionString, EventHubName);

            processor.ProcessEventAsync += ProcessEventHandler;
            processor.ProcessErrorAsync += ProcessErrorHandler;

            await processor.StartProcessingAsync();

            await Task.Delay(TimeSpan.FromSeconds(5));

            await processor.StopProcessingAsync();
        }

        private static Task ProcessErrorHandler(ProcessErrorEventArgs arg)
        {
            Console.WriteLine($"\tPartition '{ arg.PartitionId}': an unhandled exception was encountered.");
            Console.WriteLine(arg.Exception.Message);

            return Task.CompletedTask;
        }

        private static Task ProcessEventHandler(ProcessEventArgs arg)
        {
            var payload = Encoding.UTF8.GetString(arg.Data.Body.ToArray());

            var data = JsonSerializer.Deserialize<HousingData>(payload);

            var context = new MLContext();

            var model = context.Model.Load("./Model/housing-model.zip", out DataViewSchema inputSchema);

            var predictionEngine = context.Model.CreatePredictionEngine<HousingData, HousingPrediction>(model, inputSchema: inputSchema);

            var prediction = predictionEngine.Predict(data);

            Console.WriteLine($"Prediction is {prediction.PredictedHouseValue}");

            return Task.CompletedTask;
        }
    }
}
