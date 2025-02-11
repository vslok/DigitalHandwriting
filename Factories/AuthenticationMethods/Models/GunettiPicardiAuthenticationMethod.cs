using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
    internal class GunettiPicardiAuthenticationMethod : AuthenticationMethod
    {
        public GunettiPicardiAuthenticationMethod(List<double> userKeyPressedTimes, List<double> userBetweenKeysTimes, List<List<double>> userKeyPressedTimesProfile, List<List<double>> userBetweenKeysTimesProfile) : base(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile)
        {
        }

        public override bool Authenticate(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            if (n > 1)
            {
                return nGraphAuthentication(n, loginKeyPressedTimes, loginBetweenKeysTimes);
            }

            var authResult = Calculations.GunettiPicardiMetric(new Dictionary<string, List<double>>()
            {
                { "H", UserKeyPressedTimes },
                { "DU", UserBetweenKeysTimes },
            }, 
            new Dictionary<string, List<double>>()
            {
                { "H", loginKeyPressedTimes },
                { "DU", loginBetweenKeysTimes },
            }, 1.15);

            Trace.WriteLine($"auth result : {authResult}");

            // Возвращаем true, если результат аутентификации меньше порогового значения
            return authResult < 0.15;
        }

        private bool nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            var userGraph = Calculations.CalculateNGraph(n, UserKeyPressedTimes, UserBetweenKeysTimes);
            var loginGraph = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            // Вычисляем общий результат аутентификации как среднее значение трех расстояний
            var authResult = Calculations.GunettiPicardiMetric(userGraph, loginGraph, 1.15);

            Trace.WriteLine($"Auth result : {authResult}");

            return authResult < 0.15;
        }
    }
}
