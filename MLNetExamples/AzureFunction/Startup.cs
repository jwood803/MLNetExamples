using AzureFunction;
using Microsoft.Azure.Functions.Extensions.DependencyInjection;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Hosting;
using Microsoft.Extensions.ML;
using Microsoft.WindowsAzure.Storage;
using System;
using System.IO;

[assembly: FunctionsStartup(typeof(Startup))]
namespace AzureFunction
{
    class Startup : FunctionsStartup
    {
        public override void Configure(IFunctionsHostBuilder builder)
        {
            var connectionString = Environment.GetEnvironmentVariable("AzureWebJobsStorage", EnvironmentVariableTarget.Process);

            var storageAccount = CloudStorageAccount.Parse(connectionString);

            var client = storageAccount.CreateCloudBlobClient();

            var container = client.GetContainerReference("models");

            var model = container.GetBlockBlobReference("housing-model.zip");

            var uri = model.Uri.AbsoluteUri;

            builder.Services.AddPredictionEnginePool<HousingData, HousingPrediction>()
                .FromUri(uri);
        }
    }
}
