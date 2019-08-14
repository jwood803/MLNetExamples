using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Microsoft.Extensions.ML;

namespace AzureFunction
{
    public class SalaryPredict
    {
        private readonly PredictionEnginePool<SalaryData, SalaryPrediction> _predictionEnginePool;

        public SalaryPredict(PredictionEnginePool<SalaryData, SalaryPrediction> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }

        [FunctionName("SalaryPredict")]
        public async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            var salaryData = JsonConvert.DeserializeObject<SalaryData>(requestBody);

            var prediction = _predictionEnginePool.Predict(salaryData);

            return new OkObjectResult(prediction);
        }
    }
}
