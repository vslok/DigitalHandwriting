using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting
{
    public static class ConstStrings
    {
        public const string CheckText = "lorem ipsum dolor sit amet consectetur adipiscing elit";

        public const string DbName = "DigitalHandwriting.db";

        public const string FirstRegistrationStepDescription = "Придумайте проверочную фразу от 11 до 20 символов";
        public const string SecondRegistrationStepDescription = "Введите фразу снова.";
        public const string ThirdRegistrationStepDescription = "И еще разок.";
        public const string FourthRegistrationStepDescription = "И снова.";
        public const string FifthRegistrationStepDescription = "Завершите регистрацию";
    }
}
