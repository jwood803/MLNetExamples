using AzureFunction;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Hosting;
using Microsoft.Extensions.ML;
using Microsoft.WindowsAzure.Storage;
using System;
using System.IO;

[assembly: WebJobsStartup(typeof(Startup))]
namespace AzureFunction
{
    class Startup : IWebJobsStartup
    {
        public void Configure(IWebJobsBuilder builder)
        {
            var connectionString = Environment.GetEnvironmentVariable("AzureWebJobsStorage", EnvironmentVariableTarget.Process);

            var storageAccount = CloudStorageAccount.Parse(connectionString);

            var client = storageAccount.CreateCloudBlobClient();

            var container = client.GetContainerReference("models");

            var model = container.GetBlockBlobReference("salary-model.zip");

            var uri = model.Uri.AbsoluteUri;

            builder.Services.AddPredictionEnginePool<SalaryData, SalaryPrediction>()
                .FromUri(uri);
        }
    }
}
