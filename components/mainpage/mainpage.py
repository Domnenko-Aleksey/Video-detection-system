import json

def mainpage(CORE):
    CORE.debug('/components/mainpage/mainpage.py')

    CORE.addHeadFile('/templates/general/css/DAN.css')
    CORE.addHeadFile('/templates/general/js/DAN.js')
    CORE.addHeadFile('/templates/general/css/mainpage.css')
    CORE.addHeadFile('/templates/general/js/mainpage.js')

    CORE.content = f'''
        <div class="dan_flex_row_start gap_20">
            <div class="mp_panel">
                <div class="mp_panel_title">Загрузка данных</div>
                <div class="dan_flex_row_start">
                    <input id="time_treshold" type="number" class="dan_input" value="10" min="0" max="30">
                    <div class="dan_flex_row_start time_treshold_text"> Время минимального пребывания в кадре, секунд.</div>
                </div>
                <div>
                    <input id="video_file" class="dan_input" type="file">
                    <input id="button_submit" class="dan_button_red" name="submit" type="submit" value="Отправить">
                    <p class="file-return"></p>
                </div>
            </div>
            <div class="mp_panel">
                <div class="mp_panel_title">Информация</div>
                <div id="result"></div>
            </div>
        </div>
    '''
