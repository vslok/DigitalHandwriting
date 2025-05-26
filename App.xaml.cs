using System;
using System.Windows;
using DigitalHandwriting.Context;
using DigitalHandwriting.Services;
using DigitalHandwriting.Stores;
using DigitalHandwriting.ViewModels;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System.Threading.Tasks;

namespace DigitalHandwriting
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        public static IHost? AppHost { get; private set; }

        public App()
        {
            AppHost = Host.CreateDefaultBuilder()
            .ConfigureServices((hostContext, services) =>
            {
                services.AddDbContext<ApplicationContext>();

                services.AddScoped<SettingsService>();
                services.AddTransient<KeyboardMetricsCollectionService>();
                services.AddSingleton<DataMigrationService>();

                services.AddSingleton<NavigationStore>();

                services.AddSingleton<MainViewModel>();
                services.AddSingleton<UserInfoViewModel>();

                services.AddTransient<HomeViewModel>();
                services.AddTransient<RegistrationViewModel>();
                services.AddTransient<AdministrationPanelViewModel>();

                services.AddSingleton<MainWindow>((provider) =>
                {
                    return new MainWindow()
                    {
                        DataContext = provider.GetRequiredService<MainViewModel>()
                    };
                });

                services.AddTransient<Func<HomeViewModel>>((provider) =>
                {
                    return () => new HomeViewModel(
                        provider.GetRequiredService<Func<RegistrationViewModel>>(),
                        provider.GetRequiredService<NavigationStore>(),
                        provider.GetRequiredService<KeyboardMetricsCollectionService>()
                    );
                });
                services.AddTransient<Func<RegistrationViewModel>>((provider) =>
                {
                    return () => new RegistrationViewModel(
                        provider.GetRequiredService<Func<HomeViewModel>>(),
                        provider.GetRequiredService<NavigationStore>(),
                        provider.GetRequiredService<KeyboardMetricsCollectionService>()
                    );
                });
            }).Build();
        }

        protected override async void OnStartup(StartupEventArgs e)
        {
            await AppHost!.StartAsync();

            using (var scope = AppHost.Services.CreateScope())
            {
                var dbContext = scope.ServiceProvider.GetRequiredService<ApplicationContext>();
                await dbContext.Database.EnsureCreatedAsync();

                var settingsService = scope.ServiceProvider.GetRequiredService<SettingsService>();
                await settingsService.LoadSettingsAsync();
            }

            var navigationStore = AppHost.Services.GetRequiredService<NavigationStore>();
            navigationStore.CurrentViewModel = AppHost.Services.GetRequiredService<HomeViewModel>();

            var mainWindow = AppHost.Services.GetRequiredService<MainWindow>();
            mainWindow.Show();

            base.OnStartup(e);
        }

        protected override async void OnExit(ExitEventArgs e)
        {
            if (AppHost != null)
            {
                using (var scope = AppHost.Services.CreateScope())
                {
                    var context = scope.ServiceProvider.GetRequiredService<ApplicationContext>();
                    await context.DisposeAsync();
                }
                await AppHost.StopAsync();
                AppHost.Dispose();
            }
            base.OnExit(e);
        }
    }
}
