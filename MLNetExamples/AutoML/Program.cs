using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;

namespace AutoML
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var trainData = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

            var settings = new RegressionExperimentSettings
            {
                MaxExperimentTimeInSeconds = 20,
                OptimizingMetric = RegressionMetric.MeanAbsoluteError
            };

            var labelColumnInfo = new ColumnInformation()
            {
                LabelColumnName = "Label"
            };

            var progress = new Progress<RunDetail<RegressionMetrics>>(p => 
            {
                if (p.ValidationMetrics != null)
                {
                    Console.WriteLine($"Current Result - {p.TrainerName}, {p.ValidationMetrics.RSquared}, {p.ValidationMetrics.MeanAbsoluteError}");
                }
            });

            var experiment = context.Auto().CreateRegressionExperiment(settings);

            var result = experiment.Execute(trainData, labelColumnInfo, progressHandler: progress);

            Console.WriteLine(Environment.NewLine);
            Console.WriteLine("Best run:");
            Console.WriteLine($"Trainer name - {result.BestRun.TrainerName}");
            Console.WriteLine($"RSquared - {result.BestRun.ValidationMetrics.RSquared}");
            Console.WriteLine($"MAE - {result.BestRun.ValidationMetrics.MeanAbsoluteError}");

            Console.ReadLine();
        }
    }
}
