import neptune

class Neptune_Fluent:
    def mulai(enc_model, dec_model, num_pre_token=5):
        run = neptune.init_run(
            project="andialifs/fluent-tesis-playground-24",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZTY2YWQ3My04OTBkLTQ2OWUtYTc1Ni1jYjk0MGZhMWFiNGEifQ==",
        ) 

        param = {
            "encoder_model" : enc_model,
            "decoder_model" : dec_model,
            "num_pre_token" : num_pre_token
        }
        run["parameters"] = param

        return run