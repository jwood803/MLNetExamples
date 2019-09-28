using Microsoft.ML;
using System;
using System.Linq;

namespace ScaleData
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

            var columns = data.Schema
                .Select(col => col.Name)
                .Where(col => col != "Label" && col != "OceanProximity")
                .Select(col => new InputOutputColumnPair($"{col}_scaled", col))
                .ToArray();

            var scaling = context.Transforms.NormalizeMinMax(columns);

            var dataScaled = scaling.Fit(data).Transform(data);

            var preview = dataScaled.Preview();
        }
    }
}
