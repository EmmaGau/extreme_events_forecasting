class CustomModule(pl.LightningModule):
    def __init__(self, config, data_dirs, total_steps):
        super(CustomModule, self).__init__()
        self.save_hyperparameters(config)
        self.model = Conv3DEra(in_channels=1, hidden_dim=16, out_dim=1)
        self.total_steps = total_steps
        self.data_dirs = data_dirs
        self.train_config = deepcopy(config)
        self.val_config = deepcopy(config)
        self.test_config = deepcopy(config)
        
        self.train_config['dataset']['relevant_years'] = config['dataset']['relevant_years']
        self.val_config['dataset']['relevant_years'] = VAL_YEARS
        self.test_config['dataset']['relevant_years'] = TEST_YEARS
        
        self.scaler = DataScaler(config['scaler'])
        self.temp_aggregator_factory = TemporalAggregatorFactory(config['temporal_aggregator'], self.scaler)
        self.train_dataset = DatasetEra(self.train_config, data_dirs, self.temp_aggregator_factory)
        self.val_dataset = DatasetEra(self.val_config, data_dirs, self.temp_aggregator_factory)
        self.test_dataset = DatasetEra(self.test_config, data_dirs, self.temp_aggregator_factory)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=8, shuffle=True, num_workers=4)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=False, num_workers=4)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=8, shuffle=False, num_workers=4)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams['optim']['lr'], weight_decay=self.hparams['optim']['wd'])
        
        warmup_steps = int(self.hparams['optim']['warmup_percentage'] * self.total_steps)
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1., step / warmup_steps))
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(self.total_steps - warmup_steps))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

def main():
    wandb.init(project='linear_era', config=wandb_config)
    
    # Compute total steps
    total_steps = CustomModule.get_total_num_steps(len(train_dataset), 8, 5)  # Example calculation
    
    model = CustomModule(config=wandb_config, data_dirs=data_dirs, total_steps=total_steps)
    
    # Logger
    wandb_logger = WandbLogger(project='linear_era', config=wandb_config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=3, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Trainer
    trainer = pl.Trainer(max_epochs=5, logger=wandb_logger, callbacks=[checkpoint_callback, early_stop_callback, lr_monitor])
    trainer.fit(model, train_dataloaders=model.train_dataloader, val_dataloaders=model.val_dataloader)
    trainer.test(model, dataloaders=model.test_dataloader)
    
    wandb.finish()

if __name__ == "__main__":
    main()