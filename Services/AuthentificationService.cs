using DigitalHandwriting.Helpers;
using DigitalHandwriting.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace DigitalHandwriting.Services
{
    public static class AuthentificationService
    {
        public static bool PasswordAuthentification(User user, string password)
        {
            var userPasswordHash = user.Password;
            var userPasswordSalt = user.Salt;

            var inputPasswordHash = EncryptionService.GetPasswordHash(password, userPasswordSalt);

            return userPasswordHash.Equals(inputPasswordHash);
        }

        public static bool HandwritingAuthentification(User user, List<int> loginKeyPressedTimes, List<int> loginBetweenKeysTimes,
            out double keyPressedDistance, out double betweenKeysDistance)
        {
            var userKeyPressedTimes = JsonSerializer.Deserialize<List<int>>(user.KeyPressedTimes);
            var userBetweenKeysTimes = JsonSerializer.Deserialize<List<int>>(user.BetweenKeysTimes);

            keyPressedDistance = Calculations.EuclideanDistance(userKeyPressedTimes, loginKeyPressedTimes);
            betweenKeysDistance = Calculations.EuclideanDistance(userBetweenKeysTimes, loginBetweenKeysTimes);

            return keyPressedDistance <= 0.20 && betweenKeysDistance <= 0.30;
        }
    }
}
