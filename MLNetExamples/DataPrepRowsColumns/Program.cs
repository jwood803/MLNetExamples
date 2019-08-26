using Microsoft.ML;
using System;

namespace DataPrepRowsColumns
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv",
                hasHeader: true, separatorChar: ',');

            // Select columns
            var selectCols = context.Transforms.SelectColumns("HousingMedianAge", "TotalBedrooms");
            var selectColsTransform = selectCols.Fit(data).Transform(data);

            //DisplayColumns(selectColsTransform);

            // Drop columns
            var dropCols = context.Transforms.DropColumns("Latitude", "Longitude");
            var dropColsTransforms = dropCols.Fit(data).Transform(data);

            //DisplayColumns(dropColsTransforms);

            // Shuffle rows
            //DisplayColumns(data);

            //Console.WriteLine("*********************************");

            var shuffleRows = context.Data.ShuffleRows(data, seed: 42);

            //DisplayColumns(shuffleRows);

            // Take rows
            var takeRows = context.Data.TakeRows(data, 2);

            //DisplayColumns(takeRows);

            // Filter rows
            var filterRows = context.Data.FilterRowsByColumn(data, "Population",
                lowerBound: 0, upperBound: 1000);

            DisplayColumns(filterRows);

            Console.ReadLine();
        }

        private static void DisplayColumns(IDataView data)
        {
            var preview = data.Preview(maxRows: 5);

            string previewData = "";

            for (int i = 0; i < preview.RowView.Length; i++)
            {
                foreach (var item in preview.RowView[i].Values)
                {
                    previewData += $"{item.Key}: {item.Value} ";
                }

                Console.WriteLine("----------------------------------");
                Console.WriteLine(previewData);
                previewData = "";
            }
        }
    }
}
