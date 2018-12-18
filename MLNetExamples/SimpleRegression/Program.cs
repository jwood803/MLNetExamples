using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace SimpleRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            var reader = TextLoader.CreateReader(context, ctx => (
                YearsExperience: ctx.LoadFloat(0),
                Salary: ctx.LoadFloat(1)
            ), hasHeader: true, separator: ',');

            var data = reader.Read(new MultiFileSource("SalaryData.csv"));

            var pipeline = reader.MakeNewEstimator()
                .Append(r => (
                    r.Salary,
                    Prediction: context.Regression.Trainers.Sdca(label: r.Salary, features: r.YearsExperience.AsVector())
                ));

            var model = pipeline.Fit(data).AsDynamic;

            var predictionFunc = model.MakePredictionFunction<SalaryData, SalaryPrediction>(context);

            var prediction = predictionFunc.Predict(new SalaryData { YearsExperience = 8 });

            Console.WriteLine($"Predicted salary - {String.Format("{0:C}", prediction.PredictedSalary)}");

            Console.Read();
        }
    }
}
