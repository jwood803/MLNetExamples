using Microsoft.ML;
using System;

namespace SimpleRegressionUpdate
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            // Load data
            var trainData = context.Data.LoadFromTextFile<SalaryData>("./SalaryData.csv", hasHeader: true, separatorChar: ',');

            // Build model
            var pipeline = context.Transforms.Concatenate("Features", "YearsExperience")
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(trainData);

            // Evaluate
            var predictions = model.Transform(trainData);

            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine($"R^2 - {metrics.RSquared}");

            // Predict
            var newData = new SalaryData
            {
                YearsExperience = 1.1f
            };

            var predictionFunc = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(model);

            var prediction = predictionFunc.Predict(newData);

            Console.WriteLine($"Prediction - {prediction.PredictedSalary}");

            Console.ReadLine();
        }
    }
}
