function getWeatherData() {
  // API key from OpenWeatherMap
  const apiKey = '25e61362e516e536c462687854283f3e';
  
  // List of cities for which we are fetching weather data to be used in the epidemiology model
  const cities = ['Nairobi', 'Lima', 'Beijing', 'New Delhi', 'Kampala','Ol Kalou', 'Engineer', 'Njabini','Meru', 'Maua', 'Timau','Iten', 'Kapsowar','Molo', 'Njoro', 'Nakuru', 'Kuresoi','Thika', 'Limuru', 'Githunguri','Bomet', 'Sotik', 'Mulot Sunset Town','Nyeri', 'Karatina','Kerugoya', 'Kutus','Eldoret', 'Turbo', 'Moi\'s Bridge','Kitale', 'Endebess','Kapenguria', 'Makutano','Bungoma', 'Webuye','Cuzco', 'Pisac', 'Urubamba','Puno', 'Juliaca', 'Ayaviri','Huancavelica', 'Lircay', 'Pampas','Huancayo', 'Jauja', 'Tarma','Ayacucho', 'Huanta', 'San Miguel','Huaraz', 'Caraz', 'Chacas','Cajamarca', 'Celendín', 'Jaén','Arequipa', 'Chivay', 'Camaná','Abancay', 'Andahuaylas','Huaral', 'Huacho', 'Canta','Cerro de Pasco', 'Oxapampa','Trujillo', 'Otuzco', 'Huamachuco', 'Agra', 'Kannauj', 'Farrukhabad', 'Aligarh', 'Kolkata', 'Hooghly', 'Bardhaman', 'Cooch Behar', 'Patna', 'Nalanda', 'Muzaffarpur', 'Gaya','Jalandhar', 'Amritsar', 'Ludhiana', 'Hoshiarpur','Ahmedabad', 'Deesa', 'Gandhinagar', 'Banaskantha', 'Indore', 'Gwalior', 'Bhopal', 'Hoshangabad','Shimla', 'Solan', 'Mandi', 'Kullu','Dehradun', 'Nainital', 'Almora', 'Haldwani','Bengaluru', 'Hassan', 'Mysuru', 'Chikkamagaluru','Guwahati', 'Jorhat', 'Tezpur', 'Dibrugarh','Ranchi', 'Dhanbad', 'Hazaribagh', 'Bokaro','Shillong', 'Tura', 'Jowai','Hohhot', 'Baotou', 'Chifeng','Harbin', 'Qiqihar', 'Mudanjiang','Lanzhou', 'Tianshui', 'Dingxi','Kunming', 'Dali', 'Lijiang','Chengdu', 'Mianyang', 'Dazhou','Guiyang', 'Anshun', 'Zunyi','Xian', 'Baoji', 'Yulin','Chongqing', 'Wanzhou', 'Yongchuan','Yinchuan', 'Shizuishan', 'Zhongwei','Urumqi', 'Kashgar', 'Korla', 'Xining', 'Golmud','Lhasa', 'Shigatse','Kabale', 'Katuna','Kisoro', 'Bunagana','Mbale', 'Bududa','Kapchorwa', 'Suam','Kasese', 'Hima','Fort Portal', 'Kijura', 'Bundibugyo', 'Nyahuka','Mbarara', 'Isingiro','Bushenyi', 'Ishaka','Rukungiri', 'Kanungu','Ntungamo', 'Rwashamaire','Rubanda', 'Ikumba' 
];  // We can expand to more cities in the future and they can be added here in the list above

  // Loop through each city and fetch its weather data
  cities.forEach(function(city) {
    // Construct the API URL
    const url = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${apiKey}&units=metric`;

    try {
      // Fetch the weather data
      const response = UrlFetchApp.fetch(url);
      const data = JSON.parse(response.getContentText());

      // Extract weather details
      const temperature = data.main.temp;
      const humidity = data.main.humidity;
      const windSpeed = data.wind.speed;
      const date = new Date();

      // Extract precipitation (rain or snow)
      let precipitation = 0; // Default to 0 if no precipitation
      if (data.rain && data.rain['1h']) {
        precipitation = data.rain['1h']; // Rain in the last hour (mm)
      } else if (data.snow && data.snow['1h']) {
        precipitation = data.snow['1h']; // Snow in the last hour (mm)
      }
      // Extract rain and snow separately
      let rain = 0; // Default to 0 if no rain
      if (data.rain && data.rain['1h']) {
        rain = data.rain['1h']; // Rain in the last hour (mm)
      }

      let snow = 0; // Default to 0 if no snow
      if (data.snow && data.snow['1h']) {
        snow = data.snow['1h']; // Snow in the last hour (mm)
      }

      // Get the active sheet
      const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

      // Append the data to the sheet
      sheet.appendRow([date, city, temperature, humidity, windSpeed, precipitation, rain, snow]);

    } catch (e) {
      Logger.log('Error fetching data for city: ' + city + ' - ' + e.message);
    }
  });
}

function createTrigger() {
  // Create a trigger to run the getWeatherData function every day
  ScriptApp.newTrigger('getWeatherData')
    .timeBased()
    .everyDays(1)  // Run once every day
    .atHour(6)      // You can specify the hour for the update (e.g., 6 AM)
    .create();
}
