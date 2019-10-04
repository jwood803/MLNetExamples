using Microsoft.ML;
using System;

namespace SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<SentimentData>("sentiment.csv", hasHeader: true, separatorChar: ',', allowQuoting: true);

            var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            var model = pipeline.Fit(data);

            var predictionEngine = context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            var prediction = predictionEngine.Predict(new SentimentData { Text = "This is a bad movie" });

            Console.WriteLine($"Prediction - {prediction.Prediction} with score - {prediction.Score}");

            var newPrediction = predictionEngine.Predict(new SentimentData { Text = "This is the best dinner ever!" });

            Console.WriteLine($"Prediction - {newPrediction.Prediction} with score - {newPrediction.Score}");

            Console.ReadLine();
        }
    }
}
