# from openxlab.model import inference
# import openxlab


# openxlab.login(ak='pe09yeqow7qzaox85drw', sk='xyd8yblq13r2gbq5nz12wjazn5xlvzarzjmmejad', re_login=True)
# result = inference("demooooooo/tiananmen1", ['./celeba_test.png', 'bbox_mask.png'])
# print(result)
# with open("result.jpg", "wb") as f:
#     f.write(result)


from openxlab.model import inference
import openxlab

openxlab.login(ak='jy9k5vyz219r87wxp0w6', sk='oyjp5vvrblpzyl2dpykqaxnlra3zaw189g0qok6n', re_login=True)      # 请前往“账号与安全- 密钥管理”获取 AK&SK
results = inference(model_repo='liujinyao/classification', input=['./celeba_test.png']) 
print(results)    # 请将 input 参数的值更改为你的正式输入内容
with open("result.jpg", "wb") as f:
    f.write(results)