from .rope import Models as Models
from .rope import VideoManager as VM
import json
import torch
import numpy as np
import os
import folder_paths
from comfy.utils import ProgressBar
import pickle    
import subprocess
#from .videoCombine import RopeVideoCombine
from .videoCombine import ffmpeg_process
import torchaudio
from .utils import ffmpeg_path
import mimetypes
import random
from tqdm import tqdm

#####WEB_DIRECTORY = "./web"
def combine_audio_video(audio_path, video_path, output_path):       
    command = [
        ffmpeg_path,
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ]
    
    subprocess.run(command, check=True)
    return output_path

def get_mime_type(file_path):
    # 获取文件的 MIME 类型
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # 如果无法猜测类型，返回默认类型
    if mime_type is None:
        return 'application/octet-stream'    
    return mime_type

class RopeWrapper_DetectNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

                "models": ("ROPE_MODEL", ),
                #"vm": ("ROPE_VM", ),
                "input_image": ("IMAGE", ),
                #"source_face": ("IMAGE", ),
                "SimilarityThreshold":("FLOAT", {"default": 70, "min": 0.0, "max": 100, "step": 1}),
                "detection_threshold":("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                
            }
        }

    RETURN_TYPES = ("INT","DETECTRESULT","IMAGE",)
    RETURN_NAMES = ("humanCount","DETECTRESULT","foundFaces",)
    FUNCTION = "run"
    CATEGORY = "RopeWrapper"

    def findCosineDistance(self, vector1, vector2):
        cos_dist = 1.0 - np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)) # 2..0
        return 100.0-cos_dist*50.0
    
    def getEmbedding(self, img, models):
        pad_scale = 0.2
        padded_width = int(img.size()[1]*(1.+pad_scale))
        padded_height = int(img.size()[0]*(1.+pad_scale))

        padding = torch.zeros((padded_height, padded_width, 3), dtype=torch.uint8, device='cuda:0')

        width_start = int(img.size()[1]*pad_scale/2)
        width_end = width_start+int(img.size()[1])
        height_start = int(img.size()[0]*pad_scale/2)
        height_end = height_start+int(img.size()[0])

        padding[height_start:height_end, width_start:width_end,  :] = img
        img = padding

        img = img.permute(2,0,1)
        try:
            kpss = models.run_detect(img, max_num=1)[0] # Just one face here
        except IndexError:
            return None
        else:
            face_emb, cropped = models.run_recognize(img, kpss)
            return face_emb, img ,cropped
        
    def run(self, models,input_image,SimilarityThreshold,detection_threshold):

        # #print("Comfy UI Image shape:", input_image.shape)
        # try: 
        #     load_file = open("custom_nodes/Rope/saved_parameters.json", "r")
        # except FileNotFoundError:
        #     print('No save file created yet!')
        # else: 
        #         # Load the file and save it to parameters
        #     vm.parameters = json.load(load_file)
        #     load_file.close()
        
       # output_Padding=[]
       # output_cropped=[]
        # source_face_emb=None
        # source_faces=[]
        # for face_img in source_face:            
        #     tempImg = torch.round(face_img * 255)
        #     tempImg = tempImg.to(torch.uint8).to('cuda')    
        #     source_face_emb, PaddingImg,_ = self.getEmbedding(tempImg, models)
        #     if source_face_emb is not None:
        #         source_faces.append(source_face_emb)
        #         print('source_face_emb got!!!')

            # PaddingImg = PaddingImg.permute(1,2,0)
            # PaddingImg = PaddingImg.to(torch.float32)/255
            # output_Padding.append(PaddingImg)   
            # crop = cv2.cvtColor(croppedImg.cpu().numpy(), cv2.COLOR_BGR2RGB)
            # crop = cv2.resize(crop, (85, 85))
            
            # crop = crop.permute(1,2,0)
            # crop = crop.to(torch.float32)/255
            # output_cropped.append(crop)                  


        videoSwapInfo=[]
        currentFrame=0
        found_faces = []
        #output = []
        pbar = ProgressBar(len(input_image))
        for img in input_image:
            
            img = img.permute(2,0,1)
            img = torch.round(img * 255)
            img = img.to(torch.uint8).to('cuda')            
            kpss = models.run_detect(img, 'Retinaface',max_num=50,score=detection_threshold)

            kps_emb_list = []
            for face_kps in kpss:
                face_emb,cropped = models.run_recognize( img, face_kps)
                kps_emb_list.append([face_kps, face_emb,cropped])
            
           # print(f'frame: {currentFrame}, faces in this frame: {len(kpss)}, length of found_faces: {len(found_faces)}')
            
            FrameSwapInfo=[]
            faceIndexInThisFrame=0
            if kps_emb_list:
                for fface in kps_emb_list:
                    found = False
                    for found_face in found_faces:
                        sim = self.findCosineDistance(fface[1], found_face["Embedding"])
                       # print((f'sim: {sim}'))
                        if sim > SimilarityThreshold:
                            found = True
                            FrameSwapInfo.append([found_face["index"], fface[0]])
                           # print(f"swap face, index: {found_face['index']}, faceIndexInThisFrame:{faceIndexInThisFrame}")
                            break
                    if not found:
                        index=len(found_faces)
                        found_faces.append({"Face": fface[0], "Embedding": fface[1],"index":index,"Cropped":fface[2]})
                        FrameSwapInfo.append([index,fface[0]])
                       # print(f"add new face, index: {index}, faceIndexInThisFrame:{faceIndexInThisFrame}")
                    faceIndexInThisFrame+=1
            videoSwapInfo.append(FrameSwapInfo )

            # for swapInfo in FrameSwapInfo:
            #     if swapInfo[0] < len(source_faces):
            #         source_emb = source_faces[swapInfo[0]]
            #         ## temp ## img = vm.swap_core(img, swapInfo[1], source_emb, vm.parameters, vm.control)
            #         print(f'use source[{swapInfo[0]}] swap  in frame {currentFrame}')
            #img = vm.swap_core(img, kps_emb_list[0][0], source_face_emb, vm.parameters, vm.control)

            #print("shape after swap_core:",img.shape)
            ##shape after swap_core: torch.Size([3, 1706, 1279])

          #  img = img.permute(1,2,0)
            #print("shape after permute:",img.shape)
            ##shape after permute: torch.Size([1706, 1279, 3])
          #  img = img.to(torch.uint8)   

          #  img = img.to(torch.float32)/255
          #  output.append(img)
            currentFrame+=1
            pbar.update(1)

        outputFoundFaces=[]
        for face in found_faces:
            #print(f'face index: {face["index"]}, face: {face["Face"]}')
            faceImg = face["Cropped"].to(torch.float32)/255
            #print(f'faceImg shape: {faceImg.shape}')
            outputFoundFaces.append(faceImg)
            
        #tensor_stacked = torch.stack(output)
        #print("tensor_stacked shape:",tensor_stacked.shape)
        ##tensor_stacked shape: torch.Size([1, 1706, 1279, 3])

        currentHumanCount = len(found_faces)
        detectResult={"faceCount":currentHumanCount,"swapInfo":videoSwapInfo}
        # with open('detectResult.pkl', 'wb') as f:
        # pickle.dump(detectResult, f)
        #print('found_faces count:',currentHumanCount)
        return ( currentHumanCount, detectResult, torch.stack(outputFoundFaces), )
    #torch.stack(output_cropped),torch.stack(output_Padding),) 

class RopeWrapper_LoadModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("ROPE_MODEL","ROPE_VM")
    RETURN_NAMES = ("ROPE_MODEL","ROPE_VM")
    FUNCTION = "run"
    CATEGORY = "RopeWrapper"
    model=None
    vm=None
    def run(self, unique_id):
        if self.model is None:
            self.model = Models.Models()
            self.model.setModelPath(os.path.dirname(os.path.realpath(__file__))+"/")
            self.vm = VM.VideoManager(self.model)
        
        return ( self.model, self.vm )

class RopeWrapper_SwapNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "models": ("ROPE_MODEL", ),
                "vm": ("ROPE_VM", ),
                "input_image": ("IMAGE", ),
                "source_face": ("IMAGE", ),
                "detectResult": ("DETECTRESULT", ),
                "combineVideo":("BOOLEAN", {"default": False}),
                "frame_rate":("FLOAT", {"default": 30.0}),
                "filenamePrefix": ("STRING", {"default": 'Rope_', "multiline": False}),
                "saveOutput":("BOOLEAN", {"default": False}),
                "outputFrameIndex":("INT", {"default": 0}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "ROPE_Options":("ROPE_OPTION",),
                "source_target_matching": ("STRING", {"default": ' ', "multiline": True}),                
            }, 
        }

    RETURN_TYPES = ("IMAGE","STRING","STRING",)
    RETURN_NAMES = ("SwappedImage","fileName","fileNameAndPath",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "RopeWrapper"
    def getEmbedding(self, img, models):
        pad_scale = 0.2
        padded_width = int(img.size()[1]*(1.+pad_scale))
        padded_height = int(img.size()[0]*(1.+pad_scale))

        padding = torch.zeros((padded_height, padded_width, 3), dtype=torch.uint8, device='cuda:0')

        width_start = int(img.size()[1]*pad_scale/2)
        width_end = width_start+int(img.size()[1])
        height_start = int(img.size()[0]*pad_scale/2)
        height_end = height_start+int(img.size()[0])

        padding[height_start:height_end, width_start:width_end,  :] = img
        img = padding

        img = img.permute(2,0,1)
        try:
            kpss = models.run_detect(img, max_num=1)[0] # Just one face here
        except IndexError:
            return None,None,None
        else:
            face_emb, cropped = models.run_recognize(img, kpss)
            return face_emb, img ,cropped
        
    def parseString(self,str):
        try:
            if str is None:
                return None
            else:
                if str == ' ':
                    return None
                else:
                    lines = str.split(';')
                    return [ [int(x) for x in line.split(',')] for line in lines]
        except:
            return None
             
    def run(self, models,vm,input_image,source_face,detectResult,combineVideo,frame_rate,filenamePrefix,saveOutput,outputFrameIndex,audio=None,ROPE_Options=None,source_target_matching=None):
        #tempFile =os.path.join( folder_paths.output_directory, "temp.mp4")
        if saveOutput:
            output_dir = folder_paths.get_output_directory()            
        else:
            output_dir = folder_paths.get_temp_directory()   

        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filenamePrefix, output_dir)            

        filename = filename + ''.join(random.choice("abcdefghijklmnopqrstupvxyz0123456789") for x in range(5))
        tempFile = os.path.join(full_output_folder, filename)
        if ROPE_Options is None:
            try: 
                load_file = open("custom_nodes/Rope/saved_parameters.json", "r")
            except FileNotFoundError:
                print('No save file created yet!')
            else: 
                    # Load the file and save it to parameters
                vm.parameters = json.load(load_file)
                load_file.close()
        else:
            vm.parameters=ROPE_Options

        #print('vm.parameters:',vm.parameters)
        videoSwapInfo = detectResult["swapInfo"]
        targetFaceCount=detectResult["faceCount"]
        
        if(source_target_matching is not None):
            source2target=self.parseString(source_target_matching)
        else:
            source2target=None
            #print('source_target_matching is NONE')
        if source2target is None:
            #print('source2target is NONE')
            source2target=[]
            for i in range(len(source_face)):
                source2target.append([i])
        #print('source2target:',source2target)        
        source_face_emb=None
        source_faces=[]
        print("begin get embedding from source images")
        sourceFaceID=0
        for face_img in source_face:            
            tempImg = torch.round(face_img * 255)
            tempImg = tempImg.to(torch.uint8).to('cuda')    
            source_face_emb, _,_ = self.getEmbedding(tempImg, models)
            if source_face_emb is not None:
                source_faces.append(source_face_emb)
                print('source_face_emb got:', sourceFaceID)
            else:
                source_faces.append(None)
                print('source_face_emb got None', sourceFaceID)
            sourceFaceID+=1
                

        target2source=[]
        for i in range(targetFaceCount):
            found=False
            for j in range(len(source2target)):
                for index in source2target[j]:
                    if index==i:
                        target2source.append(j)
                        found=True
                        break
            if not found:
                target2source.append(-1)
                
        #width = input_image.shape[2]
        #height = input_image.shape[1]
        num_frames = len(input_image)
        pbar = ProgressBar(num_frames)
        first_image = input_image[0]
        #input_image = iter(input_image)
        NeedPad=False

        if combineVideo:
            if (first_image.shape[1] % 8) or (first_image.shape[0] % 8):
                #output frames must be padded
                to_pad = (-first_image.shape[1] % 8,
                            -first_image.shape[0] % 8)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                            to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                def pad(image):
                    #image = image.permute((2,0,1))#HWC to CHW
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded
                    #return padded.permute((1,2,0))
                #images = map(pad, images)
                new_dims = (-first_image.shape[1] % 8 + first_image.shape[1],
                            -first_image.shape[0] % 8 + first_image.shape[0])
                dimensions = f"{new_dims[0]}x{new_dims[1]}"
                print("Output images were not of valid resolution and have had padding applied：",dimensions)
                NeedPad=True
            else:
                dimensions = f"{first_image.shape[1]}x{first_image.shape[0]}"
            
            args=[ffmpeg_path, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', dimensions, '-r', str(frame_rate), '-i', '-', '-n', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19']
            video_format= {'main_pass': ['-n', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19'], 'audio_pass': ['-c:a', 'aac'], 'save_metadata': 'False', 'extension': 'mp4'}
            env=os.environ.copy()
            sp = ffmpeg_process(args,video_format,None,tempFile+".mp4",env)
            sp.send(None)
        
        currentFrame=0
        output = []
        for img in tqdm(input_image,"Swapping faces"):
        #for img in input_image:
            
            img = img.permute(2,0,1)
            img = torch.round(img * 255)
            img = img.to(torch.uint8).to('cuda')            
            #print("procedding frame:",currentFrame)
            #print("shape before swap_core:",img.shape)
            if currentFrame < len(videoSwapInfo):
                FrameSwapInfo = videoSwapInfo[currentFrame]
                for swapInfo in FrameSwapInfo:
                    targetIndex = swapInfo[0]
                    if target2source[targetIndex] != -1 and target2source[targetIndex] < len(source_faces):
                        source_emb = source_faces[target2source[targetIndex]]
                        if source_emb is not None:
                            img = vm.swap_core(img, swapInfo[1], source_emb, vm.parameters, vm.control)

            #print("shape after swap_core:",img.shape)

            if combineVideo:
                if currentFrame == outputFrameIndex:
                    outFrame =  img.permute(1,2,0)
                    outFrame = outFrame.to(torch.float32)/255
                    outFrame=outFrame.cpu()
                    output.append(outFrame)

                if NeedPad:
                    img=pad(img)
                img = img.permute(1,2,0)    
                img = img.to(torch.uint8)   
                img=img.cpu().numpy()           
                sp.send(np.ascontiguousarray(img))
                                        
            else:
                img = img.permute(1,2,0)
                img = img.to(torch.float32)/255
                img=img.cpu()
                #print("to png, img dim:",img.shape)
                output.append(img)
            
            currentFrame+=1
            pbar.update(1)
            
        if combineVideo:    
            try:
                sp.send(None) #第一次send None是告诉sp，图片已经全部输入完成
                sp.send(None) #第二次send None是结束进程。直到报StopIteration为止
            except StopIteration:
                    pass                
            ##生成视频后，用ffmpeg转成h264的例子
            ##https://discuss.streamlit.io/t/processing-video-with-opencv-and-write-it-to-disk-to-display-in-st-video/28891/2

            output_file_full_path=tempFile+".mp4"

            if audio is not None :
                ## 合并音频的代码，来自 comfyui-mix-labnodes
                output_dir = folder_paths.get_output_directory()
                # 判断是否是 Tensor 类型
                is_tensor = not isinstance(audio, dict)
                # print('#判断是否是 Tensor 类型',is_tensor,audio)
                if not is_tensor and 'waveform' in audio and 'sample_rate' in audio:
                    # {'waveform': tensor([], size=(1, 1, 0)), 'sample_rate': 44100}
                    is_tensor=True

                if "audio_path" in audio:
                    is_tensor=False
                    audio_file_path=audio["audio_path"]

                if is_tensor:
                    filename_prefix="audio_tmp"
                    audio_full_output_folder, audio_filename, counter, _, filename_prefix = folder_paths.get_save_image_path(
                        filename_prefix, 
                        folder_paths.get_temp_directory())
                    
                    filename_with_batch_num = audio_filename.replace("%batch_num%", str(1))
                    audioFileName = f"{filename_with_batch_num}_{counter:05}_.wav"
                    
                    audio_file_path=os.path.join(audio_full_output_folder, audioFileName)

                    torchaudio.save(audio_file_path, audio['waveform'].squeeze(0), audio["sample_rate"])
                    
                output_file_with_audio_path = tempFile+"audio.mp4"
                filename=filename+"audio.mp4"
                print("audio file :", audio_file_path)
                print("video file :", output_file_full_path)
                print("output file with audio :", output_file_with_audio_path)
                combine_audio_video(audio_file_path,output_file_full_path,output_file_with_audio_path)

                # # # # # # # Create audio file if input was provided
                # # # # # # output_file_with_audio_path = tempFile+"audio.mp4"
                # # # # # # filename=filename+"audio.mp4"
                # # # # # # video_format["audio_pass"] = ["-c:a", "aac"]

                # # # # # # # # FFmpeg command with audio re-encoding
                # # # # # # # #TODO: expose audio quality options if format widgets makes it in
                # # # # # # # #Reconsider forcing apad/shortest
                # # # # # # # min_audio_dur = currentFrame / frame_rate + 1
                # # # # # # # mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", tempFile+".mp4",
                # # # # # # #             "-i", "-", "-c:v", "copy"] \
                # # # # # # #             + video_format["audio_pass"] \
                # # # # # # #             + ["-af", "apad=whole_dur="+str(min_audio_dur),
                # # # # # # #                "-shortest", output_file_with_audio_path]

                # # # # # # # try:
                # # # # # # #     res = subprocess.run(mux_args, input=audio(), env=env,
                # # # # # # #                          capture_output=True, check=True)
                # # # # # # # except subprocess.CalledProcessError as e:
                # # # # # # #     raise Exception("An error occured in the ffmpeg subprocess:\n" \
                # # # # # # #             + e.stderr.decode("utf-8"))
                # # # # # # # if res.stderr:
                # # # # # # #     print(res.stderr.decode("utf-8"), end="", file=sys.stderr)
                output_file_full_path = output_file_with_audio_path
            else:
                filename=filename+".mp4"

            if saveOutput:
                previewOutputType="output"
            else:
                previewOutputType="temp"

            previews = [
                {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": previewOutputType,
                    "format": get_mime_type(filename),
                    "frame_rate": frame_rate,
                }
            ]
            #[{'filename': 'AdvancedLivePortrait_00001.mp4', 'subfolder': '', 'type': 'temp', 'format': 'video/h264-mp4', 'frame_rate': 30.0}]
            print("previews:",previews)
            return {"ui":{"gifs": previews}, "result": (torch.stack(output), filename,output_file_full_path,)}
        else:
            return (torch.stack(output), "","",)

class RopeWrapper_FaceRestore:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "models": ("ROPE_MODEL", ),
                "vm": ("ROPE_VM", ),
                "input_image": ("IMAGE", ),
                "detection_threshold":("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "combineVideo":("BOOLEAN", {"default": False}),                
                "frame_rate":("FLOAT", {"default": 30.0}),
                "filenamePrefix": ("STRING", {"default": 'RopeFaceRestore', "multiline": False}),
                "saveOutput":("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "ROPE_Options":("ROPE_OPTION",),
            },
        }

    RETURN_TYPES = ("IMAGE","STRING","STRING",)
    RETURN_NAMES = ("SwappedImage","fileName","fileNameAndPath",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "RopeWrapper"
        
    def parseString(self,str):
        try:
            if str is None:
                return None
            else:
                if str == ' ':
                    return None
                else:
                    lines = str.split(';')
                    return [ [int(x) for x in line.split(',')] for line in lines]
        except:
            return None
             
    def run(self, models,vm,input_image,detection_threshold,combineVideo,frame_rate,filenamePrefix,saveOutput,audio=None,ROPE_Options=None):
        #tempFile =os.path.join( folder_paths.output_directory, "temp.mp4")
        if saveOutput:
            output_dir = folder_paths.get_output_directory()            
        else:
            output_dir = folder_paths.get_temp_directory()   

        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filenamePrefix, output_dir)            

        filename = filename + ''.join(random.choice("abcdefghijklmnopqrstupvxyz0123456789") for x in range(5))
        tempFile = os.path.join(full_output_folder, filename)
        if ROPE_Options is None:
            try: 
                load_file = open("custom_nodes/Rope/saved_parameters.json", "r")
            except FileNotFoundError:
                print('No save file created yet!')
            else: 
                    # Load the file and save it to parameters
                vm.parameters = json.load(load_file)
                load_file.close()
        else:
            vm.parameters=ROPE_Options

        vm.parameters['RestorerSwitch'] = True
        num_frames = len(input_image)
        pbar = ProgressBar(num_frames)
        first_image = input_image[0]
        input_image = iter(input_image)
        NeedPad=False

        if combineVideo:
            if (first_image.shape[1] % 8) or (first_image.shape[0] % 8):
                #output frames must be padded
                to_pad = (-first_image.shape[1] % 8,
                            -first_image.shape[0] % 8)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                            to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                def pad(image):
                    #image = image.permute((2,0,1))#HWC to CHW
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded
                    #return padded.permute((1,2,0))
                #images = map(pad, images)
                new_dims = (-first_image.shape[1] % 8 + first_image.shape[1],
                            -first_image.shape[0] % 8 + first_image.shape[0])
                dimensions = f"{new_dims[0]}x{new_dims[1]}"
                print("Output images were not of valid resolution and have had padding applied：",dimensions)
                NeedPad=True
            else:
                dimensions = f"{first_image.shape[1]}x{first_image.shape[0]}"
            
            args=[ffmpeg_path, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', dimensions, '-r', str(frame_rate), '-i', '-', '-n', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19']
            video_format= {'main_pass': ['-n', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19'], 'audio_pass': ['-c:a', 'aac'], 'save_metadata': 'False', 'extension': 'mp4'}
            env=os.environ.copy()
            sp = ffmpeg_process(args,video_format,None,tempFile+".mp4",env)
            sp.send(None)
        
        currentFrame=0
        output = []
        for img in tqdm( input_image, desc="faceRestore", total=num_frames):
        #for img in input_image:
            
            img = img.permute(2,0,1)
            img = torch.round(img * 255)
            img = img.to(torch.uint8).to('cuda')            
            #print("procedding frame:",currentFrame)
            kpss = models.run_detect(img, 'Retinaface',max_num=50,score=detection_threshold)
            for kps in kpss:
                img = vm.core_withoutSwap(img, kps, vm.parameters, vm.control)                
            
            if combineVideo:
                if NeedPad:
                    img=pad(img)
                img = img.permute(1,2,0)    
                img = img.to(torch.uint8)   
                img=img.cpu().numpy()           
                sp.send(np.ascontiguousarray(img))
                                        
            else:
                img = img.permute(1,2,0)
                img = img.to(torch.float32)/255
                img=img.cpu()
                output.append(img)
            
            currentFrame+=1
            pbar.update(1)
            
        if combineVideo:    
            try:
                print("exporting mp4...")
                sp.send(None) #第一次send None是告诉sp，图片已经全部输入完成
                sp.send(None) #第二次send None是结束进程。直到报StopIteration为止
            except StopIteration:
                    pass                
            ##生成视频后，用ffmpeg转成h264的例子
            ##https://discuss.streamlit.io/t/processing-video-with-opencv-and-write-it-to-disk-to-display-in-st-video/28891/2

            output_file_full_path=tempFile+".mp4"

            if audio is not None :
                print("combine audio...")
                ## 合并音频的代码，来自 comfyui-mix-labnodes
                output_dir = folder_paths.get_output_directory()
                # 判断是否是 Tensor 类型
                is_tensor = not isinstance(audio, dict)
                # print('#判断是否是 Tensor 类型',is_tensor,audio)
                if not is_tensor and 'waveform' in audio and 'sample_rate' in audio:
                    # {'waveform': tensor([], size=(1, 1, 0)), 'sample_rate': 44100}
                    is_tensor=True

                if "audio_path" in audio:
                    is_tensor=False
                    audio_file_path=audio["audio_path"]

                if is_tensor:
                    filename_prefix="audio_tmp"
                    audio_full_output_folder, audio_filename, counter, _, filename_prefix = folder_paths.get_save_image_path(
                        filename_prefix, 
                        folder_paths.get_temp_directory())
                    
                    filename_with_batch_num = audio_filename.replace("%batch_num%", str(1))
                    audioFileName = f"{filename_with_batch_num}_{counter:05}_.wav"
                    
                    audio_file_path=os.path.join(audio_full_output_folder, audioFileName)

                    torchaudio.save(audio_file_path, audio['waveform'].squeeze(0), audio["sample_rate"])
                    
                output_file_with_audio_path = tempFile+"audio.mp4"
                filename=filename+"audio.mp4"
                #print("audio file :", audio_file_path)
                #print("video file :", output_file_full_path)
                #print("output file with audio :", output_file_with_audio_path)
                combine_audio_video(audio_file_path,output_file_full_path,output_file_with_audio_path)

                output_file_full_path = output_file_with_audio_path
            else:
                filename=filename+".mp4"

            if saveOutput:
                previewOutputType="output"
            else:
                previewOutputType="temp"

            previews = [
                {
                "filename": filename,
                "subfolder": subfolder,
                "type": "output" if saveOutput else "temp",
                "format": "video/h264-mp4",
                "frame_rate": frame_rate,
                }
            ]
            #[{'filename': 'AdvancedLivePortrait_00001.mp4', 'subfolder': '', 'type': 'temp', 'format': 'video/h264-mp4', 'frame_rate': 30.0}]
            print("previews:",previews)
            return {"ui":{"video": previews}, "result": (None, filename,output_file_full_path,)}
        else:
            return (torch.stack(output), "","",)

class RopeWrapper_OptionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "RestorerSwitch": ("BOOLEAN",{"default":False} ),
                "RestorerTypeTextSel": (['CF','GFPGAN', 'GPEN256', 'GPEN512'],),
                "RestorerDetTypeTextSel": (['Blend','Original','Reference'],),
                "RestorerSlider":("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                #ThresholdSlider
                "OrientSwitch": ("BOOLEAN",{"default":False} ),
                "OrientSlider":("INT", {"default": 180, "min": 0, "max": 270, "step": 90}),
                "StrengthSwitch": ("BOOLEAN",{"default":False} ),
                "StrengthSlider":("INT", {"default": 200, "min": 0, "max": 500, "step": 1}),
                "BorderTopSlider":("INT", {"default": 10, "min": 0, "max": 64, "step": 1}),
                "BorderSidesSlider":("INT", {"default": 10, "min": 0, "max": 64, "step": 1}),
                "BorderBottomSlider":("INT", {"default": 10, "min": 0, "max": 64, "step": 1}),
                "BorderBlurSlider":("INT", {"default": 10, "min": 0, "max": 64, "step": 1}),
                "DiffSwitch": ("BOOLEAN",{"default":False} ),
                "DiffSlider":("INT", {"default": 4, "min": 0.0, "max": 100, "step": 1}),
                "OccluderSwitch": ("BOOLEAN",{"default":False} ),
                "OccluderSlider":("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "FaceParserSwitch": ("BOOLEAN",{"default":False} ),
                "FaceParserSlider":("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "MouthParserSlider":("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "CLIPSwitch": ("BOOLEAN",{"default":False} ),
                "CLIPTextEntry": ("STRING", {"default": " "}),
                "CLIPSlider":("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "BlendSlider":("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "ColorSwitch": ("BOOLEAN",{"default":False} ),
                "ColorRedSlider":("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "ColorGreenSlider":("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "ColorBlueSlider":("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "ColorGammaSlider":("FLOAT", {"default": 0, "min": 0.0, "max": 2.0, "step": 0.02}),
                "FaceAdjSwitch": ("BOOLEAN",{"default":False} ),
                "KPSXSlider":("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "KPSYSlider":("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "KPSScaleSlider":("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "SwapperTypeTextSel": (['128','256', '512'],),
            },            
        }

    RETURN_TYPES = ("ROPE_OPTION",)
    FUNCTION = "run"
    CATEGORY = "RopeWrapper"
    def run(self, RestorerSwitch ,RestorerTypeTextSel ,RestorerDetTypeTextSel ,RestorerSlider ,OrientSwitch ,OrientSlider ,StrengthSwitch ,StrengthSlider ,BorderTopSlider ,BorderSidesSlider ,BorderBottomSlider ,BorderBlurSlider ,DiffSwitch ,DiffSlider ,OccluderSwitch ,OccluderSlider ,FaceParserSwitch ,FaceParserSlider ,MouthParserSlider ,CLIPSwitch ,CLIPTextEntry ,CLIPSlider ,BlendSlider ,ColorSwitch ,ColorRedSlider ,ColorGreenSlider ,ColorBlueSlider ,ColorGammaSlider ,FaceAdjSwitch ,KPSXSlider ,KPSYSlider ,KPSScaleSlider ,SwapperTypeTextSel):
        option={
            "RestorerSwitch": False,
            "RestorerTypeTextSel": "CF",
            "RestorerDetTypeTextSel": "Blend",
            "RestorerSlider": 100,
            "ThresholdSlider": 50,
            "OrientSwitch": False,
            "OrientSlider": 180,
            "StrengthSwitch": False,
            "StrengthSlider": 200,
            "BorderTopSlider": 10,
            "BorderSidesSlider": 10,
            "BorderBottomSlider": 10,
            "BorderBlurSlider": 10,
            "DiffSwitch": False,
            "DiffSlider": 4,
            "OccluderSwitch": False,
            "OccluderSlider": 0,
            "FaceParserSwitch": False,
            "FaceParserSlider": 0,
            "MouthParserSlider": 0,
            "CLIPSwitch": False,
            "CLIPTextEntry": "",
            "CLIPSlider": 50,
            "BlendSlider": 5,
            "ColorSwitch": False,
            "ColorRedSlider": 0,
            "ColorGreenSlider": -4,
            "ColorBlueSlider": 0,
            "ColorGammaSlider": 1,
            "FaceAdjSwitch": False,
            "KPSXSlider": 0,
            "KPSYSlider": 0,
            "KPSScaleSlider": 0,
            "FaceScaleSlider": 0,
            "ThreadsSlider": 5,
            "DetectTypeTextSel": "Retinaface",
            "DetectScoreSlider": 50,
            "RecordTypeTextSel": "FFMPEG",
            "VideoQualSlider": 18,
            "SwapperTypeTextSel": "128"
        }
        option["RestorerSwitch"]=RestorerSwitch 
        option["RestorerTypeTextSel"]=RestorerTypeTextSel 
        option["RestorerDetTypeTextSel"]=RestorerDetTypeTextSel 
        option["RestorerSlider"]=RestorerSlider 
        option["OrientSwitch"]=OrientSwitch 
        option["OrientSlider"]=OrientSlider 
        option["StrengthSwitch"]=StrengthSwitch 
        option["StrengthSlider"]=StrengthSlider 
        option["BorderTopSlider"]=BorderTopSlider 
        option["BorderSidesSlider"]=BorderSidesSlider 
        option["BorderBottomSlider"]=BorderBottomSlider 
        option["BorderBlurSlider"]=BorderBlurSlider 
        option["DiffSwitch"]=DiffSwitch 
        option["DiffSlider"]=DiffSlider 
        option["OccluderSwitch"]=OccluderSwitch 
        option["OccluderSlider"]=OccluderSlider 
        option["FaceParserSwitch"]=FaceParserSwitch 
        option["FaceParserSlider"]=FaceParserSlider 
        option["MouthParserSlider"]=MouthParserSlider 
        option["CLIPSwitch"]=CLIPSwitch 
        option["CLIPTextEntry"]=CLIPTextEntry 
        option["CLIPSlider"]=CLIPSlider 
        option["BlendSlider"]=BlendSlider 
        option["ColorSwitch"]=ColorSwitch 
        option["ColorRedSlider"]=ColorRedSlider 
        option["ColorGreenSlider"]=ColorGreenSlider 
        option["ColorBlueSlider"]=ColorBlueSlider 
        option["ColorGammaSlider"]=ColorGammaSlider 
        option["FaceAdjSwitch"]=FaceAdjSwitch 
        option["KPSXSlider"]=KPSXSlider 
        option["KPSYSlider"]=KPSYSlider 
        option["KPSScaleSlider"]=KPSScaleSlider 
        option["SwapperTypeTextSel"]=SwapperTypeTextSel 
        return (option,)
    

class RopeWrapper_SaveSwapInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fileName": ("STRING", {"default": 'TEMP_PKL', "multiline": False}),
                "detectResult": ("DETECTRESULT", ),                
            }, 
            
        }
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "RopeWrapper"
    
    def run(self, fileName,  detectResult):
        #print("os.path.realpath(__file__):",os.path.realpath(__file__))
        #base_path = os.path.dirname(os.path.realpath(__file__))
        models_dir = folder_paths.models_dir
        ROPE_MODELS_PATH = os.path.join(models_dir, "rope")
        if not os.path.exists(ROPE_MODELS_PATH):
            os.makedirs(ROPE_MODELS_PATH)
        counter = 1
        saveFile = os.path.join(ROPE_MODELS_PATH, fileName)
        while os.path.exists(saveFile):
            saveFile = os.path.join(ROPE_MODELS_PATH, fileName + '_' + str(counter))
            counter += 1
        print("Saving SwapInfo to file: ", saveFile)
        #print(detectResult) 

        with open(saveFile, 'wb') as f:
            pickle.dump(detectResult, f)       
        return fileName

class RopeWrapper_LoadSwapInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fileName": ("STRING", {"default": 'TEMP_PKL', "multiline": False}),           
            },             
        }
    RETURN_TYPES = ("DETECTRESULT",)
    RETURN_NAMES = ("DETECTRESULT",)
    FUNCTION = "run"
    CATEGORY = "RopeWrapper"
    def run(self, fileName):
        models_dir = folder_paths.models_dir
        ROPE_MODELS_PATH = os.path.join(models_dir, "rope")
        saveFile = os.path.join(ROPE_MODELS_PATH, fileName)        
        #print("Loading SwapInfo from file: ", saveFile)
        detectResult=None
        try:
            with open(saveFile, 'rb') as f:
                detectResult = pickle.load(f)      
            #print("Loaded SwapInfo from file: ", detectResult) 
            return (detectResult,)   
        except Exception as e:
            print("Error loading SwapInfo from file: ", e)
            return (None,)
        

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    #"RopeVideoCombine":RopeVideoCombine,
    "RopeWrapper_DetectNode":RopeWrapper_DetectNode,
    "RopeWrapper_LoadModels":RopeWrapper_LoadModels,
    "RopeWrapper_SwapNode":RopeWrapper_SwapNode,
    #"RopeWrapper_SwapNodeTEST":RopeWrapper_SwapNodeTEST,
    "RopeWrapper_OptionNode":RopeWrapper_OptionNode,  
    "RopeWrapper_SaveSwapInfo":RopeWrapper_SaveSwapInfo,  
    "RopeWrapper_LoadSwapInfo":RopeWrapper_LoadSwapInfo,  
    "RopeWrapper_FaceRestore":RopeWrapper_FaceRestore,
      
    
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
     #"RopeVideoCombine":"RopeVideoCombine",
     "RopeWrapper_DetectNode":"RopeWrapper_DetectNode",
     "RopeWrapper_LoadModels":"RopeWrapper_LoadModels",
     "RopeWrapper_SwapNode":"RopeWrapper_SwapNode",
     #"RopeWrapper_SwapNodeTEST":"RopeWrapper_SwapNodeTEST",
     "RopeWrapper_OptionNode":"RopeWrapper_OptionNode", 
     "RopeWrapper_SaveSwapInfo":"RopeWrapper_SaveSwapInfo",  
     "RopeWrapper_LoadSwapInfo":"RopeWrapper_LoadSwapInfo",  
     "RopeWrapper_FaceRestore":"RopeWrapper_FaceRestore",
 
}
