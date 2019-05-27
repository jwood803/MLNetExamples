using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;

namespace ExportToOnnx
{
    class Program
    {
        private static readonly string MODEL_NAME = "model.onnx";

        static void Main(string[] args)
        {
            var context = new MLContext();

            var textLoader = context.Data.CreateTextLoader(new[]
            {
                new TextLoader.Column("YearsExperience", DataKind.Single, 0),
                new TextLoader.Column("Salary", DataKind.Single, 1),
            },
            hasHeader: true,
            separatorChar: ',');

            var data = textLoader.Load("./SalaryData.csv");

            var trainTestData = context.Data.TrainTestSplit(data);

            var pipeline = context.Transforms.Concatenate("Features", "YearsExperience")
                .Append(context.Regression.Trainers.Sdca(labelColumnName: "Salary"));

            ITransformer model = pipeline.Fit(trainTestData.TrainSet);

            using (var stream = File.Create(MODEL_NAME))
            {
                context.Model.ConvertToOnnx(model, data, stream);
            }

            var estimator = context.Transforms.ApplyOnnxModel(MODEL_NAME);

            var newModel = estimator.Fit(trainTestData.TestSet);
        }
    }
}
