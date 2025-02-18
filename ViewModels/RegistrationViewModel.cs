using DigitalHandwriting.Commands;
using DigitalHandwriting.Context;
using DigitalHandwriting.Helpers;
using DigitalHandwriting.Models;
using DigitalHandwriting.Repositories;
using DigitalHandwriting.Services;
using DigitalHandwriting.Stores;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using System.Windows.Input;

namespace DigitalHandwriting.ViewModels
{
    public class RegistrationViewModel : BaseViewModel
    {
        public ICommand OnCheckTextBoxKeyDownEventCommand { get; set; }
        public ICommand OnCheckTextBoxKeyUpEventCommand { get; set; }
        public ICommand OnResetRegistrationWindowButtonClickCommand { get; set; }

        public ICommand OnRegistrationStartButtonClickCommand { get; set; }
        public ICommand OnRegistrationFinalizeButtonClickCommand { get; set; }
        public ICommand ReturnHomeCommand { get; set; }


        private int _checkTextCurrentLetterIndex = 0;

        private int _registrationStep = 0;

        private bool _isLoginBoxEnabled = true;

        private bool _isCheckTextBoxEnabled = false;

        private bool _isFinalizeButtonVisible = false;

        private bool _isStartRegistrationbuttonVisible = false;

        private string _checkTextWithUpperCase => UserPassPhrase.ToUpper();

        private string _userCheckText = "";

        private string _userLogin = "";

        private string _userPassPhrase = "";

        private readonly UserRepository _userRepository;

        private readonly KeyboardMetricsCollectionService _keyboardMetricsCollector;

        public RegistrationViewModel(
            Func<HomeViewModel> homeViewModelFactory,
            NavigationStore navigationStore,
            KeyboardMetricsCollectionService keyboardMetricsCollector)
        {
            OnCheckTextBoxKeyDownEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyDownEvent);
            OnCheckTextBoxKeyUpEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyUpEvent);
            OnRegistrationFinalizeButtonClickCommand = new NavigateCommand<HomeViewModel>(
                new NavigationService<HomeViewModel>(
                    navigationStore,
                    () => homeViewModelFactory()), OnRegistrationFinalizeButtonClick);
            ReturnHomeCommand = new NavigateCommand<HomeViewModel>(
                new NavigationService<HomeViewModel>(
                    navigationStore,
                    () => homeViewModelFactory()));
            OnResetRegistrationWindowButtonClickCommand = new Command(OnResetRegistrationWindowButtonClick);
            OnRegistrationStartButtonClickCommand = new Command(OnRegistrationStartButtonClick);

            _keyboardMetricsCollector = keyboardMetricsCollector;

            _userRepository = new UserRepository();
        }

        public bool IsLoginBoxEnabled
        {
            get => _isLoginBoxEnabled;
            set => SetProperty(ref _isLoginBoxEnabled, value);
        }

        public bool IsCheckTextBoxEnabled
        {
            get => _isCheckTextBoxEnabled;
            set => SetProperty(ref _isCheckTextBoxEnabled, value);
        }

        public bool IsPasswordTextBoxVisible => !IsCheckTextBoxEnabled && RegistrationStep == 0;

        public bool IsFinalizeButtonVisible
        {
            get => _isFinalizeButtonVisible;
            set => SetProperty(ref _isFinalizeButtonVisible, value);
        }

        public bool IsRegistrationStartButtonVisible
        {
            get => _isStartRegistrationbuttonVisible;
            set => SetProperty(ref _isStartRegistrationbuttonVisible, value);
        }


        public int RegistrationStep
        {
            get => _registrationStep;
            set
            {
                SetProperty(ref _registrationStep, value);
                if (value == 4)
                {
                    IsCheckTextBoxEnabled = false;
                    IsFinalizeButtonVisible = true;
                }
            }
        }

        public string UserLogin
        {
            get => _userLogin;
            set
            {
                SetProperty(ref _userLogin, value);
            }
        }

        public string UserPassPhrase
        {
            get => _userPassPhrase;
            set
            {
                SetProperty(ref _userPassPhrase, value);
                if (value.Length >= 20)
                {
                    IsRegistrationStartButtonVisible = true;
                    InvokeOnPropertyChangedEvent(nameof(_isStartRegistrationbuttonVisible));
                }
            }
        }

        public string UserCheckText
        {
            get => _userCheckText;
            set 
            {
                IsLoginBoxEnabled = false;
                SetProperty(ref _userCheckText, value);
            }
        }

        private void OnRegistrationFinalizeButtonClick()
        {
            User user = new User()
            {
                Login = UserLogin,
                FirstH = JsonSerializer.Serialize(_keyboardMetricsCollector.KeyPressedTimes[0]),
                SecondH = JsonSerializer.Serialize(_keyboardMetricsCollector.KeyPressedTimes[1]),
                ThirdH = JsonSerializer.Serialize(_keyboardMetricsCollector.KeyPressedTimes[2]),
                FirstUD = JsonSerializer.Serialize(_keyboardMetricsCollector.BetweenKeysTimes[0]),
                SecondUD = JsonSerializer.Serialize(_keyboardMetricsCollector.BetweenKeysTimes[1]),
                ThirdUD = JsonSerializer.Serialize(_keyboardMetricsCollector.BetweenKeysTimes[2]),
                Password = EncryptionService.GetPasswordHash(UserPassPhrase, out string salt),
                Salt = salt
            };

            _userRepository.AddUser(user);
        }

        private void OnCheckTextBoxKeyUpEvent(object props)
        {
            var e = (KeyEventArgs)props;

            Trace.WriteLine($"{e.Key} = keyUp");

            _keyboardMetricsCollector.OnKeyUpEvent(e);

            if (_checkTextCurrentLetterIndex == _checkTextWithUpperCase.Length && UserCheckText.Length == _checkTextWithUpperCase.Length)
            {
                _keyboardMetricsCollector.IncreaseMetricsCollectingStep(_checkTextWithUpperCase);
                RegistrationStep++;
                UserCheckText = "";
                _checkTextCurrentLetterIndex = 0;
            }
        }

        private void OnCheckTextBoxKeyDownEvent(object props)
        {
            var e = (KeyEventArgs)props;
            Trace.WriteLine($"{e.Key} = keyDown");
            if (!Helper.CheckCurrentLetterKeyPressed(e, _checkTextCurrentLetterIndex, _checkTextWithUpperCase) || _registrationStep == 4)
            {
                e.Handled = true;
                return;
            }

            _keyboardMetricsCollector.OnKeyDownEvent(e);
            _checkTextCurrentLetterIndex++;
        }

        private void OnRegistrationStartButtonClick()
        {
            RegistrationStep++;
            IsRegistrationStartButtonVisible = false;
            IsCheckTextBoxEnabled = true;
            InvokeOnPropertyChangedEvent(nameof(IsPasswordTextBoxVisible));
        }

        private void OnResetRegistrationWindowButtonClick()
        {
            RegistrationStep = 0;
            UserLogin = "";
            UserCheckText = "";
            UserPassPhrase = "";
            _checkTextCurrentLetterIndex = 0;
            _keyboardMetricsCollector.ResetMetricsCollection();
            IsLoginBoxEnabled = true;
            IsCheckTextBoxEnabled = false;
            IsFinalizeButtonVisible = false;
            IsRegistrationStartButtonVisible = false;
            InvokeOnPropertyChangedEvent(nameof(IsPasswordTextBoxVisible));
        }
    }
}
