using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace SimpleRegression
{
    public class SalaryData
    {
        [Column("0")]
        public float YearsExperience;

        [Column("1", name: "Label")]
        public float Salary;
    }
}
