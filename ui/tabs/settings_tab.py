import shutil
import os
import gradio as gr
import roop.globals
import ui.globals

available_themes = ["Default", "gradio/glass", "gradio/monochrome", "gradio/seafoam", "gradio/soft", "gstaff/xkcd", "freddyaboulton/dracula_revamped", "ysharma/steampunk"]
image_formats = ['jpg','png', 'webp']
video_formats = ['avi','mkv', 'mp4', 'webm']
video_codecs = ['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']
providerlist = None

settings_controls = []

def get_available_gpus():
    """Returns a list of available GPU devices"""
    print("get_available_gpus() called")
    gpu_list = []
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"Found {device_count} CUDA devices")
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_list.append(f"GPU {i}: {gpu_name}")
        if not gpu_list:
            gpu_list = ["No GPU detected"]
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        gpu_list = ["No GPU detected"]
    print(f"Returning GPU list: {gpu_list}")
    return gpu_list

def get_current_gpu_value(gpu_list):
    """Safely get the current GPU value for the dropdown"""
    try:
        if roop.globals.cuda_device_id < len(gpu_list) and "GPU" in gpu_list[roop.globals.cuda_device_id]:
            return gpu_list[roop.globals.cuda_device_id]
    except:
        pass
    return gpu_list[0] if gpu_list else "No GPU detected"

def settings_tab():
    from roop.core import suggest_execution_providers
    global providerlist

    
    providerlist = suggest_execution_providers()
    with gr.Tab("⚙ Settings"):
        with gr.Row():
            with gr.Column():
                themes = gr.Dropdown(available_themes, label="Theme", info="Change needs complete restart", value=roop.globals.CFG.selected_theme)
            with gr.Column():
                settings_controls.append(gr.Checkbox(label="Public Server", value=roop.globals.CFG.server_share, elem_id='server_share', interactive=True))
                settings_controls.append(gr.Checkbox(label='Clear output folder before each run', value=roop.globals.CFG.clear_output, elem_id='clear_output', interactive=True))
                output_template = gr.Textbox(label="Filename Output Template", info="(file extension is added automatically)", lines=1, placeholder='{file}_{time}', value=roop.globals.CFG.output_template)
            with gr.Column():
                input_server_name = gr.Textbox(label="Server Name", lines=1, info="Leave blank to run locally", value=roop.globals.CFG.server_name)
            with gr.Column():
                input_server_port = gr.Number(label="Server Port", precision=0, info="Leave at 0 to use default", value=roop.globals.CFG.server_port)
        with gr.Row():
            with gr.Column():
                settings_controls.append(gr.Dropdown(providerlist, label="Provider", value=roop.globals.CFG.provider, elem_id='provider', interactive=True))
                
                gpu_list = get_available_gpus()
                gpu_value = get_current_gpu_value(gpu_list)
                gpu_dropdown = gr.Dropdown(choices=gpu_list, label="Select GPU", value=gpu_value, elem_id='gpu_device', interactive=True, visible=True)
                
                chk_det_size = gr.Checkbox(label="Use default Det-Size", value=True, elem_id='default_det_size', interactive=True)
                settings_controls.append(gr.Checkbox(label="Force CPU for Face Analyser", value=roop.globals.CFG.force_cpu, elem_id='force_cpu', interactive=True))
                max_threads = gr.Slider(1, 32, value=roop.globals.CFG.max_threads, label="Max. Number of Threads", info='default: 3', step=1.0, interactive=True)
            with gr.Column():
                memory_limit = gr.Slider(0, 128, value=roop.globals.CFG.memory_limit, label="Max. Memory to use (Gb)", info='0 meaning no limit', step=1.0, interactive=True)
                settings_controls.append(gr.Dropdown(image_formats, label="Image Output Format", info='default: png', value=roop.globals.CFG.output_image_format, elem_id='output_image_format', interactive=True))
            with gr.Column():
                settings_controls.append(gr.Dropdown(video_codecs, label="Video Codec", info='default: libx264', value=roop.globals.CFG.output_video_codec, elem_id='output_video_codec', interactive=True))
                settings_controls.append(gr.Dropdown(video_formats, label="Video Output Format", info='default: mp4', value=roop.globals.CFG.output_video_format, elem_id='output_video_format', interactive=True))
                video_quality = gr.Slider(0, 100, value=roop.globals.CFG.video_quality, label="Video Quality (crf)", info='default: 14', step=1.0, interactive=True)
            with gr.Column():
                with gr.Group():
                    settings_controls.append(gr.Checkbox(label='Use OS temp folder', value=roop.globals.CFG.use_os_temp_folder, elem_id='use_os_temp_folder', interactive=True))
                    settings_controls.append(gr.Checkbox(label='Show video in browser (re-encodes output)', value=roop.globals.CFG.output_show_video, elem_id='output_show_video', interactive=True))
                button_apply_restart = gr.Button("Restart Server", variant='primary')
                button_clean_temp = gr.Button("Clean temp folder")
                button_apply_settings = gr.Button("Apply Settings")

    chk_det_size.select(fn=on_option_changed)

    # Settings
    for s in settings_controls:
        s.select(fn=on_settings_changed)
    max_threads.input(fn=lambda a,b='max_threads':on_settings_changed_misc(a,b), inputs=[max_threads])
    memory_limit.input(fn=lambda a,b='memory_limit':on_settings_changed_misc(a,b), inputs=[memory_limit])
    video_quality.input(fn=lambda a,b='video_quality':on_settings_changed_misc(a,b), inputs=[video_quality])
    gpu_dropdown.select(fn=on_gpu_changed)

    # button_clean_temp.click(fn=clean_temp, outputs=[bt_srcfiles, input_faces, target_faces, bt_destfiles])
    button_clean_temp.click(fn=clean_temp)
    button_apply_settings.click(apply_settings, inputs=[themes, input_server_name, input_server_port, output_template])
    button_apply_restart.click(restart)


def on_option_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.value)
            return
    raise gr.Error(f'Unhandled Setting for {evt.target}')


def on_settings_changed_misc(new_val, attribname):
    if hasattr(roop.globals.CFG, attribname):
        setattr(roop.globals.CFG, attribname, new_val)
    else:
        print("Didn't find attrib!")
        

def on_gpu_changed(evt: gr.SelectData):
    """Handle GPU selection change"""
    try:
        gpu_string = evt.value
        if gpu_string and "GPU" in gpu_string:
            # Extract GPU index from "GPU 0: NVIDIA GeForce..."
            gpu_id = int(gpu_string.split(":")[0].split("GPU")[1].strip())
            roop.globals.cuda_device_id = gpu_id
            if hasattr(roop.globals.CFG, 'gpu_device_id'):
                roop.globals.CFG.gpu_device_id = gpu_id
                roop.globals.CFG.save()
            print(f'GPU {gpu_id} selected')
            gr.Info(f'GPU {gpu_id} selected. Restart may be required for changes to take effect.')
    except Exception as e:
        print(f"Error changing GPU: {e}")
        gr.Warning('Failed to parse GPU selection')


def on_settings_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.value)
            return
            
    raise gr.Error(f'Unhandled Setting for {evt.target}')

def clean_temp():
    from ui.main import prepare_environment
    
    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs = []
    if not roop.globals.CFG.use_os_temp_folder:
        shutil.rmtree(os.environ["TEMP"])
    prepare_environment()
    gr.Info('Temp Files removed')
    return None,None,None,None


def apply_settings(themes, input_server_name, input_server_port, output_template):
    from ui.main import show_msg

    roop.globals.CFG.selected_theme = themes
    roop.globals.CFG.server_name = input_server_name
    roop.globals.CFG.server_port = input_server_port
    roop.globals.CFG.output_template = output_template
    roop.globals.CFG.save()
    show_msg('Settings saved')


def restart():
    ui.globals.ui_restart_server = True
