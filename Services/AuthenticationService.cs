using DigitalHandwriting.Helpers;
using DigitalHandwriting.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace DigitalHandwriting.Services
{
    public class HandwritingAuthenticationResult
    {
        public double KeyPressedDistance { get; set; }

        public double BetweenKeysDistance { get; set; }

        public double BetweenKeysPressDistance { get; set; }

        public double AuthenticationResult { get; set; }
    }

    public static class AuthenticationService
    {
        public static bool PasswordAuthentication(User user, string password)
        {
            var userPasswordHash = user.Password;
            var userPasswordSalt = user.Salt;

            var inputPasswordHash = EncryptionService.GetPasswordHash(password, userPasswordSalt);

            return userPasswordHash.Equals(inputPasswordHash);
        }

        public static bool HandwritingAuthentication(
            User user, 
            List<int> loginKeyPressedTimes, 
            List<int> loginBetweenKeysTimes,
            List<int> loginBetweenKeysPressTimes,
            out double keyPressedDistance, 
            out double betweenKeysDistance,
            out double betweenKeysPressDistance)
        {
            var userKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimesMedians);
            var userBetweenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysTimesMedians);
            var userBetweenKeysPressTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysPressTimesMedians);

            if (userKeyPressedTimes == null || userBetweenKeysTimes == null || userBetweenKeysPressTimes == null)
            {
                throw new Exception("Incorrect user authentication parameters in db");
            }

            var userKeyPressedNormalized = Calculations.Normalize(userKeyPressedTimes);
            var loginKeyPressedNormalized = Calculations.Normalize(loginKeyPressedTimes);

            var userBetweenKeysNormalized = Calculations.Normalize(userBetweenKeysTimes);
            var loginBetweenKeysNormalized = Calculations.Normalize(loginBetweenKeysTimes);

            var userBetweenKeysPressNormalized = Calculations.Normalize(userBetweenKeysPressTimes);
            var loginBetweenKeysPressNormalized = Calculations.Normalize(loginBetweenKeysPressTimes);

            var keyPressedDistanceError = CalculateErrors(userKeyPressedNormalized, loginKeyPressedNormalized, 0.2);
            var keyBetweenKeysDistanceError = CalculateErrors(userBetweenKeysNormalized, loginBetweenKeysNormalized, 0.2);
            var keyBetweenKeysPressDistanceError = CalculateErrors(userBetweenKeysPressNormalized, loginBetweenKeysPressNormalized, 0.2);

            keyPressedDistance = keyPressedDistanceError / (double)userKeyPressedTimes.Count;
            betweenKeysDistance = keyBetweenKeysDistanceError / (double)userBetweenKeysTimes.Count;
            betweenKeysPressDistance = keyBetweenKeysPressDistanceError / (double)userBetweenKeysPressTimes.Count;

            var authResult = (keyPressedDistanceError + keyBetweenKeysDistanceError + keyBetweenKeysPressDistanceError) 
                / (double)(userKeyPressedTimes.Count + userBetweenKeysTimes.Count + userBetweenKeysPressTimes.Count);

            Trace.WriteLine($"Key pressed errors: {keyPressedDistanceError}, " +
                $"Between keys errors: {keyBetweenKeysDistanceError}, " +
                $"Between keys press errors: {keyBetweenKeysPressDistanceError} " +
                $"auth result : {authResult}");

            //keyPressedDistance = Calculations.ManhattanDistance(userKeyPressedTimes, loginKeyPressedTimes);
            //betweenKeysDistance = Calculations.ManhattanDistance(userBetweenKeysTimes, loginBetweenKeysTimes);

            return authResult < 0.15;
        }

        private static int CalculateErrors(List<double> userData, List<double> loginData, double threshold)
        {
            int errors = 0;
            for (int i = 0; i < userData.Count; i++)
            {
                var result = Calculations.ManhattanDistance(userData[i], loginData[i]);
                if (result > threshold)
                {
                    errors++;
                }
            }
            return errors;
        }

        private static bool EuclidianAuthentication(User user, List<int> loginKeyPressedTimes, List<int> loginBetweenKeysTimes, List<int> loginBetweenKeysPressTimes)
        {
            var userKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimesMedians);
            var userBetweenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysTimesMedians);
            var userBetweenKeysPressTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysPressTimesMedians);

            if (userKeyPressedTimes == null || userBetweenKeysTimes == null || userBetweenKeysPressTimes == null)
            {
                throw new Exception("Incorrect user authentication parameters in db");
            }

            var userKeyPressedTimesDouble = userKeyPressedTimes.ConvertAll(x => (double)x);
            var loginKeyPressedTimesDouble = loginKeyPressedTimes.ConvertAll(x => (double)x);

            var userBetweenKeysTimesDouble = userBetweenKeysTimes.ConvertAll(x => (double)x);
            var loginBetweenKeysTimesDouble = loginBetweenKeysTimes.ConvertAll(x => (double)x);

            var userBetweenKeysPressTimesDouble = userBetweenKeysPressTimes.ConvertAll(x => (double)x);
            var loginBetweenKeysPressTimesDouble = loginBetweenKeysPressTimes.ConvertAll(x => (double)x);

            // Вычисляем Евклидово расстояние для каждого из трех параметров
            var keyPressedDistance = Calculations.EuclideanDistance(userKeyPressedTimesDouble, loginKeyPressedTimesDouble);
            var betweenKeysDistance = Calculations.EuclideanDistance(userBetweenKeysTimesDouble, loginBetweenKeysTimesDouble);
            var betweenKeysPressDistance = Calculations.EuclideanDistance(userBetweenKeysPressTimesDouble, loginBetweenKeysPressTimesDouble);

            // Вычисляем общий результат аутентификации как среднее значение трех расстояний
            var authResult = (keyPressedDistance + betweenKeysDistance + betweenKeysPressDistance) / 3.0;

            Trace.WriteLine($"Key pressed distance: {keyPressedDistance}, " +
                $"Between keys distance: {betweenKeysDistance}, " +
                $"Between keys press distance: {betweenKeysPressDistance} " +
                $"auth result : {authResult}");

            // Возвращаем true, если результат аутентификации меньше порогового значения
            return authResult < 0.15;
        }

        public static bool EuclidianAuthenticationWithNormalization(User user, List<int> loginKeyPressedTimes, 
            List<int> loginBetweenKeysTimes, List<int> loginBetweenKeysPressTimes)
        {
            var userKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimesMedians);
            var userBetweenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysTimesMedians);
            var userBetweenKeysPressTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysPressTimesMedians);

            if (userKeyPressedTimes == null || userBetweenKeysTimes == null || userBetweenKeysPressTimes == null)
            {
                throw new Exception("Incorrect user authentication parameters in db");
            }

            // Нормализация данных
            var userKeyPressedNormalized = Calculations.Normalize(userKeyPressedTimes);
            var loginKeyPressedNormalized = Calculations.Normalize(loginKeyPressedTimes);

            var userBetweenKeysNormalized = Calculations.Normalize(userBetweenKeysTimes);
            var loginBetweenKeysNormalized = Calculations.Normalize(loginBetweenKeysTimes);

            var userBetweenKeysPressNormalized = Calculations.Normalize(userBetweenKeysPressTimes);
            var loginBetweenKeysPressNormalized = Calculations.Normalize(loginBetweenKeysPressTimes);

            // Вычисляем Евклидово расстояние для каждого из трех параметров
            var keyPressedDistance = Calculations.EuclideanDistance(userKeyPressedNormalized, loginKeyPressedNormalized);
            var betweenKeysDistance = Calculations.EuclideanDistance(userBetweenKeysNormalized, loginBetweenKeysNormalized);
            var betweenKeysPressDistance = Calculations.EuclideanDistance(userBetweenKeysPressNormalized, loginBetweenKeysPressNormalized);

            // Вычисляем общий результат аутентификации как среднее значение трех расстояний
            var authResult = (keyPressedDistance + betweenKeysDistance + betweenKeysPressDistance) / 3.0;

            Trace.WriteLine($"Key pressed distance: {keyPressedDistance}, " +
                $"Between keys distance: {betweenKeysDistance}, " +
                $"Between keys press distance: {betweenKeysPressDistance} " +
                $"auth result : {authResult}");

            // Возвращаем true, если результат аутентификации меньше порогового значения
            return authResult < 0.15;
        }

        public static bool ManhattanAuthentication(User user, List<int> loginKeyPressedTimes, List<int> loginBetweenKeysTimes, List<int> loginBetweenKeysPressTimes)
        {
            var userKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimesMedians);
            var userBetweenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysTimesMedians);
            var userBetweenKeysPressTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysPressTimesMedians);

            if (userKeyPressedTimes == null || userBetweenKeysTimes == null || userBetweenKeysPressTimes == null)
            {
                throw new Exception("Incorrect user authentication parameters in db");
            }

            var userKeyPressedTimesDouble = userKeyPressedTimes.ConvertAll(x => (double)x);
            var loginKeyPressedTimesDouble = loginKeyPressedTimes.ConvertAll(x => (double)x);

            var userBetweenKeysTimesDouble = userBetweenKeysTimes.ConvertAll(x => (double)x);
            var loginBetweenKeysTimesDouble = loginBetweenKeysTimes.ConvertAll(x => (double)x);

            var userBetweenKeysPressTimesDouble = userBetweenKeysPressTimes.ConvertAll(x => (double)x);
            var loginBetweenKeysPressTimesDouble = loginBetweenKeysPressTimes.ConvertAll(x => (double)x);

            // Вычисляем Манхэттенское расстояние для каждого из трех параметров
            var keyPressedDistance = Calculations.ManhattanDistance(userKeyPressedTimesDouble, loginKeyPressedTimesDouble);
            var betweenKeysDistance = Calculations.ManhattanDistance(userBetweenKeysTimesDouble, loginBetweenKeysTimesDouble);
            var betweenKeysPressDistance = Calculations.ManhattanDistance(userBetweenKeysPressTimesDouble, loginBetweenKeysPressTimesDouble);

            // Вычисляем общий результат аутентификации как среднее значение трех расстояний
            var authResult = (keyPressedDistance + betweenKeysDistance + betweenKeysPressDistance) / 3.0;

            Trace.WriteLine($"Key pressed distance: {keyPressedDistance}, " +
                $"Between keys distance: {betweenKeysDistance}, " +
                $"Between keys press distance: {betweenKeysPressDistance} " +
                $"auth result : {authResult}");

            // Возвращаем true, если результат аутентификации меньше порогового значения
            return authResult < 0.15;
        }

        public static bool ManhattanAuthenticationWithNormalization(User user, List<int> loginKeyPressedTimes, 
            List<int> loginBetweenKeysTimes, List<int> loginBetweenKeysPressTimes)
        {
            var userKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimesMedians);
            var userBetweenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysTimesMedians);
            var userBetweenKeysPressTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysPressTimesMedians);

            if (userKeyPressedTimes == null || userBetweenKeysTimes == null || userBetweenKeysPressTimes == null)
            {
                throw new Exception("Incorrect user authentication parameters in db");
            }

            var userKeyPressedNormalized = Calculations.Normalize(userKeyPressedTimes);
            var loginKeyPressedNormalized = Calculations.Normalize(loginKeyPressedTimes);

            var userBetweenKeysNormalized = Calculations.Normalize(userBetweenKeysTimes);
            var loginBetweenKeysNormalized = Calculations.Normalize(loginBetweenKeysTimes);

            var userBetweenKeysPressNormalized = Calculations.Normalize(userBetweenKeysPressTimes);
            var loginBetweenKeysPressNormalized = Calculations.Normalize(loginBetweenKeysPressTimes);

            // Вычисляем Манхэттенское расстояние для каждого из трех параметров
            var keyPressedDistance = Calculations.ManhattanDistance(userKeyPressedNormalized, loginKeyPressedNormalized);
            var betweenKeysDistance = Calculations.ManhattanDistance(userBetweenKeysNormalized, loginBetweenKeysNormalized);
            var betweenKeysPressDistance = Calculations.ManhattanDistance(userBetweenKeysPressNormalized, loginBetweenKeysPressNormalized);

            // Вычисляем общий результат аутентификации как среднее значение трех расстояний
            var authResult = (keyPressedDistance + betweenKeysDistance + betweenKeysPressDistance) / 3.0;

            Trace.WriteLine($"Key pressed distance: {keyPressedDistance}, " +
                $"Between keys distance: {betweenKeysDistance}, " +
                $"Between keys press distance: {betweenKeysPressDistance} " +
                $"auth result : {authResult}");

            // Возвращаем true, если результат аутентификации меньше порогового значения
            return authResult < 0.15;
        }

        public static bool ITADAuthentication(User user, List<int> loginKeyPressedTimes,
            List<int> loginBetweenKeysTimes, List<int> loginBetweenKeysPressTimes)
        {
            var userFirstKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimesFirst);
            var userFirstBetweenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysTimesFirst);
            var userFirstBetweenKeysPressTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysPressTimesFirst);

            var userSecondKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimesSecond);
            var userSecondBetweenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysTimesSecond);
            var userSecondBetweenKeysPressTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysPressTimesSecond);

            var userThirdKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimesThird);
            var userThirdBetweenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysTimesThird);
            var userThirdBetweenKeysPressTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysPressTimesThird);

            var loginKeyPressedTimesDouble = loginKeyPressedTimes.ConvertAll(x => (double)x);
            var loginBetweenKeysTimesDouble = loginBetweenKeysTimes.ConvertAll(x => (double)x);
            var loginBetweenKeysPressTimesDouble = loginBetweenKeysPressTimes.ConvertAll(x => (double)x);

            var userFirstKeyPressedTimesDouble = userFirstKeyPressedTimes.ConvertAll(x => (double)x);
            var userFirstBetweenKeysTimesDouble = userFirstBetweenKeysTimes.ConvertAll(x => (double)x);
            var userFirstBetweenKeysPressTimesDouble = userFirstBetweenKeysPressTimes.ConvertAll(x => (double)x);

            var userSecondKeyPressedTimesDouble = userSecondKeyPressedTimes.ConvertAll(x => (double)x);
            var userSecondBetweenKeysTimesDouble = userSecondBetweenKeysTimes.ConvertAll(x => (double)x);
            var userSecondBetweenKeysPressTimesDouble = userSecondBetweenKeysPressTimes.ConvertAll(x => (double)x);

            var userThirdKeyPressedTimesDouble = userThirdKeyPressedTimes.ConvertAll(x => (double)x);
            var userThirdBetweenKeysTimesDouble = userThirdBetweenKeysTimes.ConvertAll(x => (double)x);
            var userThirdBetweenKeysPressTimesDouble = userThirdBetweenKeysPressTimes.ConvertAll(x => (double)x);


            // Вычисляем Манхэттенское расстояние для каждого из трех параметров
            var keyPressedDistance = Calculations.ITAD(loginKeyPressedTimesDouble, new List<List<double>>()
            {
                userFirstKeyPressedTimesDouble, userSecondKeyPressedTimesDouble, userThirdKeyPressedTimesDouble,
            });
            var betweenKeysDistance = Calculations.ITAD(loginBetweenKeysTimesDouble, new List<List<double>>()
            {
                userFirstBetweenKeysTimesDouble, userSecondBetweenKeysTimesDouble, userThirdBetweenKeysTimesDouble,
            });
            var betweenKeysPressDistance = Calculations.ITAD(loginBetweenKeysPressTimesDouble, new List<List<double>>()
            {
                userFirstBetweenKeysPressTimesDouble, userSecondBetweenKeysPressTimesDouble, userThirdBetweenKeysPressTimesDouble,
            });

            // Вычисляем общий результат аутентификации как среднее значение трех расстояний
            var authResult = (keyPressedDistance + betweenKeysDistance + betweenKeysPressDistance) / 3.0;

            Trace.WriteLine($"Key pressed distance: {keyPressedDistance}, " +
                $"Between keys distance: {betweenKeysDistance}, " +
                $"Between keys press distance: {betweenKeysPressDistance} " +
                $"auth result : {authResult}");

            return authResult > 0.45;
        }

        public static bool GunettiPicardiMetricAuthentication(User user, List<int> loginKeyPressedTimes,
            List<int> loginBetweenKeysTimes, List<int> loginBetweenKeysPressTimes)
    }
}
