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
    public class HousingPredict
    {
        private readonly PredictionEnginePool<HousingData, HousingPrediction> _predictionEnginePool;

        public HousingPredict(PredictionEnginePool<HousingData, HousingPrediction> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }

        [FunctionName("HousingPredict")]
        public async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            var housingData = JsonConvert.DeserializeObject<HousingData>(requestBody);

            var prediction = _predictionEnginePool.Predict(housingData);

            return new OkObjectResult(prediction);
        }
    }
}
