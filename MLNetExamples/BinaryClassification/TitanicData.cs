﻿using Microsoft.ML.Runtime.Api;

namespace BinaryClassification
{
    public class TitanicData
    {
        [Column("0")]
        public float PassengerId;

        [Column("1", name: "Label")]
        public bool HasSurvived;

        [Column("2")]
        public float Pclass;

        [Column("3")]
        public string Name;

        [Column("4")]
        public string Sex;

        [Column("5")]
        public float Age;

        [Column("6")]
        public float SibSp;

        [Column("7")]
        public float Parch;

        [Column("8")]
        public string Ticket;

        [Column("9")]
        public float Fare;

        [Column("10")]
        public string Cabin;

        [Column("11")]
        public string Embarked;
    }
}
