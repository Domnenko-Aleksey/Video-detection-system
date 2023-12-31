window.addEventListener('DOMContentLoaded', function(){
	IMPORT_DATA.init();
});


IMPORT_DATA = {
    // Инициализация
    init() {       
        DAN.$('button_submit').onclick = IMPORT_DATA.send_form;
    },


    // Сохранение настроек
    send_form() {
        let video_name = DAN.$('video_file').value;
        if (video_name == '') {
            alert('Не выбран файл');
            return;
        }

        let video_file = DAN.$('video_file').files[0];

        let spinner = IMPORT_DATA.spinner();
        let load_html = 
                '<div style="text-align:center;">Загрузка файла</div>' +
                '<div style="position:relative;height:100px;">' + spinner + '</div>';
        DAN.$('result').innerHTML = load_html;
	
        let form = new FormData();
		form.append('video_file', video_file);
		DAN.ajax('/video_upload_ajax', form, function(data) {
            console.log(data);
            if (data.answer == 'success') {
                console.log('PREDICT')
                IMPORT_DATA.model_predict();
            } else {
                DAN.$('result').innerHTML = 'Проблема при загрузке модели.'
            }
		})
    },


    // Предсказание модели
    model_predict() {
        let spinner = IMPORT_DATA.spinner();
        let load_html = 
                '<div style="text-align:center;color:#00ff00;">Обработка файла</div>' +
                '<div style="position:relative;height:100px;">' + spinner + '</div>';
        DAN.$('result').innerHTML = load_html;
        let time_treshold = DAN.$('time_treshold').value;
        let form = new FormData();
		form.append('time_treshold', time_treshold);
        DAN.ajax('/model_predict_ajax', form, function(data) {
            console.log(data)
            if (data.answer == 'success') {
                let rnd = Math.floor(Math.random() * 1000000);
                DAN.$('result').innerHTML = '<video class="predict_video" src="/files/predict.webm?' + rnd + '" autoplay="" controls="" loop=""></video>';
            } else {
                DAN.$('result').innerHTML = 'Проблема при обработки моделиы';
            }
		})
    },


    spinner() {
        return '<svg id="dan_spinner" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M11.501 4.025v-4.025h1v4.025l-.5-.025-.5.025zm-7.079 5.428l-3.884-1.041-.26.966 3.881 1.04c.067-.331.157-.651.263-.965zm5.995-5.295l-1.039-3.878-.967.259 1.041 3.883c.315-.106.635-.197.965-.264zm-6.416 7.842l.025-.499h-4.026v1h4.026l-.025-.501zm2.713-5.993l-2.846-2.845-.707.707 2.846 2.846c.221-.251.457-.487.707-.708zm-1.377 1.569l-3.48-2.009-.5.866 3.484 2.012c.15-.299.312-.591.496-.869zm13.696.607l3.465-2-.207-.36-3.474 2.005.216.355zm.751 1.993l3.873-1.038-.129-.483-3.869 1.037.125.484zm-3.677-5.032l2.005-3.472-.217-.125-2.002 3.467.214.13zm-1.955-.843l1.037-3.871-.16-.043-1.038 3.873.161.041zm3.619 2.168l2.835-2.834-.236-.236-2.834 2.833.235.237zm-9.327-1.627l-2.011-3.484-.865.5 2.009 3.479c.276-.184.568-.346.867-.495zm-4.285 8.743l-3.88 1.04.26.966 3.884-1.041c-.106-.314-.197-.634-.264-.965zm11.435 5.556l2.01 3.481.793-.458-2.008-3.478c-.255.167-.522.316-.795.455zm3.135-2.823l3.477 2.007.375-.649-3.476-2.007c-.116.224-.242.439-.376.649zm-1.38 1.62l2.842 2.842.59-.589-2.843-2.842c-.187.207-.383.403-.589.589zm2.288-3.546l3.869 1.037.172-.644-3.874-1.038c-.049.218-.102.434-.167.645zm.349-2.682l.015.29-.015.293h4.014v-.583h-4.014zm-6.402 8.132l1.039 3.879.967-.259-1.041-3.884c-.315.106-.635.197-.965.264zm-1.583.158l-.5-.025v4.025h1v-4.025l-.5.025zm-5.992-2.712l-2.847 2.846.707.707 2.847-2.847c-.25-.22-.487-.456-.707-.706zm-1.165-1.73l-3.485 2.012.5.866 3.48-2.009c-.185-.278-.347-.57-.495-.869zm2.734 3.106l-2.01 3.481.865.5 2.013-3.486c-.299-.149-.591-.311-.868-.495zm1.876.915l-1.042 3.886.967.259 1.04-3.881c-.33-.067-.65-.158-.965-.264z"></path></svg>';
    }
}