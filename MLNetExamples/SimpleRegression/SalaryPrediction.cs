using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace SimpleRegression
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary;
    }
}
