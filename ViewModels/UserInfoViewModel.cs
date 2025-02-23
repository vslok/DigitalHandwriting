using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using System.Collections.Generic;

namespace DigitalHandwriting.ViewModels
{
    public class UserInfoViewModel : BaseViewModel
    {
        private bool _isAuthenticated = false;

        private double _keyPressedMetric = 0.0;

        private double _betweenKeysMetric = 0.0;

        private double _betweenKeysPressMetric = 0.0;

        private double _betweenKeysResolveMetric = 0.0;

        public UserInfoViewModel(AuthenticationResult authenticationResult)
        {
            IsAuthentificated = authenticationResult.IsAuthenticated;
            KeyPressedMetric = authenticationResult.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.H);
            BetweenKeysMetric = authenticationResult.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.UD);
            BetweenKeysPressMetric = authenticationResult.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.DD);
            BetweenKeysResolveMetric = authenticationResult.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.UU);
        }

        public bool IsAuthentificated
        {
            get => _isAuthenticated;
            set => SetProperty(ref _isAuthenticated, value);
        }

        public double KeyPressedMetric
        {
            get => _keyPressedMetric;
            set => SetProperty(ref _keyPressedMetric, value);
        }

        public double BetweenKeysMetric
        {
            get => _betweenKeysMetric;
            set => SetProperty(ref _betweenKeysMetric, value);
        }

        public double BetweenKeysPressMetric
        {
            get => _betweenKeysPressMetric;
            set => SetProperty(ref _betweenKeysPressMetric, value);
        }

        public double BetweenKeysResolveMetric
        {
            get => _betweenKeysResolveMetric;
            set => SetProperty(ref _betweenKeysResolveMetric, value);
        }
    }
}
