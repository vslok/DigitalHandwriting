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

        /*        public const string FirstRegistrationStepDescription = "Для регистрации введите проверочный текст. " +
                    "Возможно ввести только текущий символ текста. " +
                    "После окончания ввода текста поле ввода очистится и будет показан следующий шаг.";*/

        public const string FirstRegistrationStepDescription = "Придумайте проверочную фразу не менее 20 символов";
        public const string SecondRegistrationStepDescription = "Введите фразу снова.";
        public const string ThirdRegistrationStepDescription = "И снова.";
        public const string FourthRegistrationStepDescription = "И еще разок.";
        public const string FifthRegistrationStepDescription = "Завершите регистрацию";
    }
}
