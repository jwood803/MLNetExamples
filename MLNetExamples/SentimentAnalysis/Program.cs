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

            Console.WriteLine($"Prediction - {prediction.Prediction} with probability - {prediction.Probability}");

            var newPrediction = predictionEngine.Predict(new SentimentData { Text = "TWTR may close at a low today." });

            Console.WriteLine($"Prediction - {newPrediction.Prediction} with probability - {newPrediction.Probability}");

            var anotherPrediction = predictionEngine.Predict(new SentimentData { Text = "TSLA is at an all time high." });

            switch (anotherPrediction.Probability)
            {
                case float p when p < .5:
                    Console.WriteLine($"TSLA sentiment is negative with probability {p}");
                    break;
                case float p when p >= .5 && p <= .7:
                    Console.WriteLine($"TSLA sentiment is neutral with probability {p}");
                    break;
                case float p when p > .7:
                    Console.WriteLine($"TSLA sentiment is positive with probability {p}");
                    break;
                default:
                    break;
            }
        }
    }
}
