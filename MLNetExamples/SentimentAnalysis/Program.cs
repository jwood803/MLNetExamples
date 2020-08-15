using Microsoft.ML;
using System;

namespace SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<SentimentData>("stock_data.csv", hasHeader: true, separatorChar: ',', allowQuoting: true);

            var pipeline = context.Transforms.Expression("Label", "(x) => x == 1 ? true : false", "Sentiment")
                .Append(context.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text)))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            var model = pipeline.Fit(data);

            var predictionEngine = context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            var prediction = predictionEngine.Predict(new SentimentData { Text = "I would buy MSFT shares." });

            Console.WriteLine($"Prediction - {prediction.Prediction} with score - {prediction.Score}");

            var newPrediction = predictionEngine.Predict(new SentimentData { Text = "TWTR may close at a low today." });

            Console.WriteLine($"Prediction - {newPrediction.Prediction} with score - {newPrediction.Score}");

            Console.ReadLine();
        }
    }
}
